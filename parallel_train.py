"""Linux云端接管现有串行队列；保留当前子任务，最多同时训练两项。"""
from pathlib import Path
import argparse
from datetime import datetime
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cloud_training", ROOT / "02_训练流程/04_训练模型/train.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
RUNTIME = training.RUNTIME


def key(job):
    return training.job_dir(job).name


def candidates(cfg, task, name):
    return [{"task": task, "model": name, "lr": lr, "seed": cfg["seeds"][0]} for lr in cfg["learning_rates"]]


def ready_jobs(cfg, lookup):
    """每个模型选参完成后才能补种子；选参只读验证成绩。"""
    jobs, selected = [], {}
    for task in cfg["tasks"]:
        for name in cfg["models"]:
            trials = candidates(cfg, task, name)
            jobs.extend(trials)
            results = [lookup(job) for job in trials]
            if all(row is not None for row in results):
                best = min(results, key=lambda row: row["best_val_RMSE"])
                selected[f"{task}/{name}"] = {"task": task, "model": name, "lr": best["lr"],
                                               "selection_seed": cfg["seeds"][0], "best_val_RMSE": best["best_val_RMSE"]}
                jobs.extend({"task": task, "model": name, "lr": best["lr"], "seed": seed} for seed in cfg["seeds"][1:])
    return jobs, selected


def alive(pid):
    path = Path(f"/proc/{pid}/status")
    try:
        return "State:\tZ" not in path.read_text()
    except FileNotFoundError:
        return False


def job_from_process(pid):
    args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    if "--job" not in args or Path(f"/proc/{pid}/cwd").resolve() != ROOT:
        raise RuntimeError(f"进程{pid}不是本工程训练子任务")
    return json.loads(args[args.index("--job") + 1])


def started_at(pid, job):
    metadata = training.job_dir(job) / "job.json"
    if metadata.exists():
        return datetime.fromisoformat(training.read(metadata)["started_at"]).timestamp()
    # 子进程可能仍在导入PyTorch，任务信息尚未写出；使用内核记录的启动时间。
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
    return boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")


def worker(job, signature):
    cfg, audit = training.configuration(), training.prepare()
    if training.fingerprint() != signature:
        raise ValueError("训练源码或数据发生变化，拒绝混用结果")
    # 不修改原训练源码；只将本进程的进度文件放到独立目录。
    training.RUNTIME = RUNTIME / "parallel_workers" / key(job)
    training.RUNTIME.mkdir(parents=True, exist_ok=True)
    training.train_job(cfg, audit, job, signature)


def schedule(queue_pid, dry_run=False):
    cfg = training.configuration()
    if cfg["ablations"]:
        raise ValueError("本入口仅用于当前无消融的七模型任务")
    signature, completed = training.fingerprint(), {}

    def lookup(job):
        job_key = key(job)
        if job_key not in completed:
            result = training.complete(job, signature)
            if result is not None:
                completed[job_key] = result
        return completed.get(job_key)

    jobs, selected = ready_jobs(cfg, lookup)
    for job in jobs:
        lookup(job)
    if dry_run:
        return {"fingerprint": signature, "completed": len(completed), "max_parallel_jobs": 2,
                "ready_pending_jobs": [job for job in jobs if lookup(job) is None]}
    if sys.platform != "linux":
        raise RuntimeError("接管现有队列仅支持Linux服务器")
    previous = RUNTIME / "parallel_status.json"
    if previous.exists():
        raise RuntimeError("已有并行接管记录，先检查状态，不能重复启动")
    args = Path(f"/proc/{queue_pid}/cmdline").read_bytes().decode().split("\0")
    if "--stage" not in args or args[args.index("--stage") + 1] != "all" or Path(f"/proc/{queue_pid}/cwd").resolve() != ROOT:
        raise RuntimeError("待接管的进程不是本工程完整队列")
    active, stopped, failure = {}, False, None

    def status(state):
        entries = []
        for entry in active.values():
            progress_path = RUNTIME / "worker.json" if entry["adopted"] else RUNTIME / "parallel_workers" / key(entry["job"]) / "worker.json"
            progress = training.read(progress_path) if progress_path.exists() else {}
            entries.append({"job": entry["job"], "pid": entry["pid"], "adopted": entry["adopted"],
                            "progress_file": str(progress_path), "progress": progress})
        value = {"status": state, "updated_at": training.now(), "pid": os.getpid(), "original_queue_pid": queue_pid,
                 "original_queue_paused": stopped, "max_parallel_jobs": 2, "completed_jobs": len(completed),
                 "total_jobs": training.job_count(cfg), "active_jobs": entries, "fingerprint": signature}
        if failure:
            value["error"] = failure
        training.write_json(previous, value)
        training.write_json(RUNTIME / "status.json", value)
        return value

    try:
        # 只挂起调度父进程；已经开始的GPU子任务继续执行。
        os.kill(queue_pid, signal.SIGSTOP)
        stopped = True
        deadline = time.monotonic() + 5
        while "State:\tT" not in Path(f"/proc/{queue_pid}/status").read_text():
            if time.monotonic() > deadline:
                raise RuntimeError("原队列未进入挂起状态")
            time.sleep(.05)
        children = Path(f"/proc/{queue_pid}/task/{queue_pid}/children").read_text().split()
        for child in children:
            pid = int(child)
            if alive(pid):
                job = job_from_process(pid)
                if lookup(job) is None:
                    started = started_at(pid, job)
                    active[key(job)] = {"job": job, "pid": pid, "adopted": True, "started": started}
        if len(active) > 1:
            raise RuntimeError("原队列存在多个子任务，拒绝重复接管")
        status("running_parallel")
        while True:
            for job_key, entry in list(active.items()):
                process = entry.get("process")
                finished = process.poll() is not None if process else not alive(entry["pid"])
                if not finished and time.time() - entry["started"] > cfg["job_timeout_seconds"]:
                    os.kill(entry["pid"], signal.SIGKILL)
                    training.write_json(training.job_dir(entry["job"]) / "failure.json", {"reason": "hard_timeout", "job": entry["job"], "updated_at": training.now()})
                    finished = True
                if finished:
                    if process:
                        process.wait()
                        entry["log"].close()
                    if lookup(entry["job"]) is None:
                        failure = f"训练失败，未自动重试：{job_key}"
                        training.write_json(training.job_dir(entry["job"]) / "failure.json", {"reason": "worker_exit", "job": entry["job"], "updated_at": training.now()})
                    del active[job_key]
            jobs, selected = ready_jobs(cfg, lookup)
            training.write_json(RUNTIME / "selection.json", {"fingerprint": signature, "models": selected})
            if not failure:
                for job in jobs:
                    if len(active) >= 2:
                        break
                    if key(job) in active or lookup(job) is not None:
                        continue
                    dest = training.job_dir(job)
                    if (dest / "epochs.csv").exists() or (dest / "failure.json").exists():
                        raise RuntimeError(f"存在未完成记录，拒绝覆盖：{dest}")
                    dest.mkdir(parents=True, exist_ok=True)
                    log = (dest / "console.log").open("w", encoding="utf-8")
                    process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                                "--job", json.dumps(job), "--signature", signature],
                                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    active[key(job)] = {"job": job, "pid": process.pid, "process": process, "log": log,
                                        "adopted": False, "started": time.time()}
                    print(f"PARALLEL TRAIN {job} pid={process.pid}", flush=True)
            current = status("draining_after_failure" if failure else "running_parallel")
            if not active:
                if failure:
                    raise RuntimeError(failure)
                if len(completed) != training.job_count(cfg):
                    raise RuntimeError("训练数量未达到49次，拒绝进入最终测试")
                # 原队列逐项校验并复用已完成记录，随后统一测试、出表和画图。
                os.kill(queue_pid, signal.SIGCONT)
                stopped = False
                return status("training_finished_handed_back_for_evaluation")
            time.sleep(2)
    except Exception as error:
        failure = str(error)
        status("failed_parallel")
        # 已有子任务继续保留；原队列仍挂起，避免异常后重复分配训练。
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--signature")
    args = parser.parse_args()
    if args.job:
        worker(json.loads(args.job), args.signature)
    else:
        print(json.dumps(schedule(args.queue_pid, args.dry_run), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
