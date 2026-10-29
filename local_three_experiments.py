"""三个本地验证实验。只新增独立入口，复用原数据、模型和训练规则。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("baseline_training", ROOT / "02_训练流程/04_训练模型/train.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import numpy as np
import pandas as pd
import torch
from adapters import NativeForecast

OUT = ROOT / "03_实验结果/06_本地原因验证"
ORIGINAL_RESULTS = base.RESULTS
ORIGINAL_BUILD = base.build
ORIGINAL_RECEIPT = base.read(ORIGINAL_RESULTS / "运行记录/fit_complete.json")
VARIANT = "TimeXer_LastAnchor"


class LastAnchor(NativeForecast):
    """只把目标输出的窗口均值加回，替换成最后实测值加回；主干不变。"""

    def __init__(self, metadata, cfg, device):
        super().__init__("TimeXer", metadata["total_channels"], cfg, metadata["time_feature_indices"])
        self.to(device)

    def forward(self, x):
        # x 是训练段标准化后的历史；整个修正在相同标准化坐标内完成。
        correction = x[:, -1, -1:] - x[:, :, -1].mean(dim=1, keepdim=True)
        return super().forward(x) + correction


def experiment_build(name, metadata, cfg, device):
    return LastAnchor(metadata, cfg, device) if name == VARIANT else ORIGINAL_BUILD(name, metadata, cfg, device)


def dump(path, value):
    base.write_json(path, value)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def environment():
    return {"python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": sys.platform}


def signatures():
    return {"baseline": base.fingerprint(), "experiment_script_sha256": base.sha256(__file__)}


def signature():
    return hashlib.sha256(json.dumps(signatures(), sort_keys=True).encode()).hexdigest()


def setup_context():
    cfg = base.configuration()
    audit = base.prepare()
    if base.fingerprint() != ORIGINAL_RECEIPT["fingerprint"]:
        raise RuntimeError("原实验源文件与权重版本不匹配，停止而不修改原实验")
    return cfg, audit


def groups(pack, thresholds):
    delta = np.abs(pack["y"] - pack["persistence"])
    return {"all": np.ones(len(delta), dtype=bool), "stable": delta <= thresholds[0],
            "moderate": (delta > thresholds[0]) & (delta <= thresholds[1]),
            "fast": delta > thresholds[1]}


def score_rows(pred, pack, model, seed, split, thresholds, **extra):
    rows = []
    for group, mask in groups(pack, thresholds).items():
        row = {"model": model, "seed": seed, "split": split, "group": group, **extra}
        row.update(base.metrics(pred[mask], pack["y"][mask], pack["persistence"][mask]))
        row["bias_m"] = float((pred[mask] - pack["y"][mask]).mean())
        rows.append(row)
    return rows


def original_job(name, seed):
    return {"task": "A", "model": name, "seed": seed,
            "lr": ORIGINAL_RECEIPT["selected"][f"A/{name}"]["lr"]}


def original_dir(job):
    return ORIGINAL_RESULTS / "01_对比实验/runs" / f"A_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def interface_check(cfg, audit):
    device = base.setup(cfg, 42)
    meta = audit["tasks"]["A"]
    x = torch.tensor(base.load("A", "train", 64)["x"], device=device)
    baseline = ORIGINAL_BUILD("TimeXer", meta, cfg, device)
    changed = LastAnchor(meta, cfg, device)
    changed.load_state_dict(baseline.state_dict(), strict=True)
    assert baseline.state_dict().keys() == changed.state_dict().keys()
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in changed.parameters())
    baseline.eval()
    changed.eval()
    before = x.clone()
    with torch.no_grad():
        a, b = baseline(x), changed(x)
        correction = x[:, -1, -1:] - x[:, :, -1].mean(1, keepdim=True)
        torch.testing.assert_close(b, a + correction, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        # 网络 head 输出为零时，原版还原到历史均值，变体还原到最后实测值。
        for model in (baseline, changed):
            model.network.head.linear.weight.zero_()
            model.network.head.linear.bias.zero_()
        torch.testing.assert_close(baseline(x), x[:, :, -1].mean(1, keepdim=True), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(changed(x), x[:, -1, -1:], rtol=1e-6, atol=1e-6)
    trainable = LastAnchor(meta, cfg, device)
    trainable.train()
    loss = trainable(x).square().mean()
    loss.backward()
    gradients = [p.grad for p in trainable.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    dump(OUT / "interface_check.json", {"passed": True, "parameters": sum(p.numel() for p in changed.parameters()),
          "unchanged_state_dict_structure": True, "formula_verified": True, "input_not_modified": True,
          "zero_head_mean_vs_last_verified": True, "finite_parameter_gradients": True})
    del baseline, changed, trainable, x
    torch.cuda.empty_cache()


def diagnose():
    cfg, audit = setup_context()
    OUT.mkdir(parents=True, exist_ok=True)
    packs = {split: base.load("A", split) for split in ("train", "val")}
    thresholds = np.quantile(np.abs(packs["train"]["y"] - packs["train"]["persistence"]), [.5, .9])
    dump(OUT / "plan.json", {"created_at": base.now(), "signatures": signatures(), "environment": environment(),
         "config": cfg, "thresholds_m": thresholds.tolist(), "threshold_source": "train labels only, q50/q90",
         "experiment_1": "35 original best checkpoints; eval-mode physical-scale train/val errors and regimes",
         "experiment_2": "frozen TimeXer, val only; 3 grouped whole-window permutations x 10 repeats x 5 seeds",
         "experiment_3": "local original TimeXer and LastAnchor, each 3 LR seed42 candidates + 4 additional seeds",
         "selection": "seed42 validation RMSE only; no test until both model selections/training complete",
         "test_status": "existing test previously inspected, new results exploratory",
         "output_formula": "native_prediction - historical_target_mean + last_observed_target",
         "unchanged": ["original source files", "protocol.json", "original weights/results", "data/scalers", "backbone/parameters", "training/selection budget"]})
    interface_check(cfg, audit)
    mean, std = base.scale("A")
    device = base.setup(cfg, 42)
    tensors = {split: torch.tensor(pack["x"], device=device) for split, pack in packs.items()}
    rows, reproduction = [], []
    for split, pack in packs.items():
        rows += score_rows(pack["persistence"], pack, "Persistence", None, split, thresholds)
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            job = original_job(name, seed)
            dest = original_dir(job)
            result = base.read(dest / "result.json")
            assert base.sha256(dest / "best.pt") == result["checkpoint_sha256"]
            base.setup(cfg, seed)
            model = ORIGINAL_BUILD(name, audit["tasks"]["A"], cfg, device)
            model.load_state_dict(torch.load(dest / "best.pt", map_location=device, weights_only=True))
            for split, pack in packs.items():
                pred = base.predict(model, tensors[split], cfg["batch_size"]) * std + mean
                rows += score_rows(pred, pack, name, seed, split, thresholds)
                if split == "val":
                    error = abs(base.metrics(pred, pack["y"], pack["persistence"])["RMSE"] - result["best_val_RMSE"])
                    reproduction.append({"model": name, "seed": seed, "val_RMSE_difference_m": error})
            del model
            print(f"EXP1 {name} seed={seed}", flush=True)
        save_csv(OUT / "experiment1_eval_metrics.csv", rows)
    save_csv(OUT / "checkpoint_reproduction.csv", reproduction)
    if max(row["val_RMSE_difference_m"] for row in reproduction) > 1e-5:
        raise RuntimeError("本地复现与云端验证误差差异超过0.01mm，需要检查数值环境")
    permutation_rows = []
    meta = audit["tasks"]["A"]
    calendar = meta["time_feature_indices"]
    feature_groups = {"upstream_physical": [i for i in range(meta["up_channels"]) if i not in calendar],
                      "local_physical": list(range(meta["up_channels"], meta["total_channels"])), "calendar": calendar}
    permutations = [np.random.default_rng(8100 + repeat).permutation(len(packs["val"]["y"])) for repeat in range(10)]
    val_x = tensors["val"]
    for seed in cfg["seeds"]:
        base.setup(cfg, seed)
        model = ORIGINAL_BUILD("TimeXer", meta, cfg, device)
        model.load_state_dict(torch.load(original_dir(original_job("TimeXer", seed)) / "best.pt", map_location=device, weights_only=True))
        clean = base.predict(model, val_x, cfg["batch_size"]) * std + mean
        clean_rmse = base.metrics(clean, packs["val"]["y"], packs["val"]["persistence"])["RMSE"]
        for group, indices in feature_groups.items():
            for repeat, permutation in enumerate(permutations):
                altered = val_x.clone()
                donor = val_x[torch.tensor(permutation, device=device)]
                altered[:, :, indices] = donor[:, :, indices]
                prediction = base.predict(model, altered, cfg["batch_size"]) * std + mean
                scores = base.metrics(prediction, packs["val"]["y"], packs["val"]["persistence"])
                permutation_rows.append({"seed": seed, "group": group, "repeat": repeat,
                       "clean_RMSE": clean_rmse, "delta_RMSE": scores["RMSE"] - clean_rmse,
                       "prediction_shift_RMS_m": float(np.sqrt(np.mean((prediction-clean)**2))), **scores})
            print(f"EXP2 seed={seed} group={group}", flush=True)
            save_csv(OUT / "experiment2_permutation.csv", permutation_rows)
        del model
    dump(OUT / "diagnostics_complete.json", {"completed_at": base.now(), "baseline_fingerprint": base.fingerprint(),
          "experiment1_checkpoint_count": len(reproduction), "experiment2_runs": len(permutation_rows),
          "test_set_used": False, "max_reproduction_difference_m": max(row["val_RMSE_difference_m"] for row in reproduction)})


def worker(job):
    cfg, audit = setup_context()
    base.RESULTS = OUT / "experiment3"
    base.RUNTIME = OUT / "workers" / f"{job['model']}_{job['seed']}_{job['lr']:g}"
    base.RUNTIME.mkdir(parents=True, exist_ok=True)
    base.build = experiment_build
    base.train_job(cfg, audit, job, signature())


def local_dir(job):
    return OUT / "experiment3/01_对比实验/runs" / f"A_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def run_one(job, cfg):
    path = local_dir(job)
    if (path / "result.json").exists():
        result = base.read(path / "result.json")
        if result["fingerprint"] != signature() or base.sha256(path / "best.pt") != result["checkpoint_sha256"]:
            raise RuntimeError(f"本地实验版本不匹配：{path}")
        return result
    if (path / "epochs.csv").exists():
        raise RuntimeError(f"存在未完成训练，不覆盖：{path}")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "console.log").open("w", encoding="utf-8") as handle:
        command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "--stage", "worker", "--job", json.dumps(job)]
        proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        print(f"TRAIN pid={proc.pid} {job}", flush=True)
        try:
            code = proc.wait(timeout=cfg["job_timeout_seconds"])
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            dump(path / "failure.json", {"reason": "hard_timeout", "job": job})
            raise RuntimeError(f"训练超时：{job}")
        if code:
            dump(path / "failure.json", {"reason": "worker_exit", "code": code, "job": job})
            raise RuntimeError(f"训练失败：{path / 'console.log'}")
    return base.read(path / "result.json")


def branch(name):
    cfg, _ = setup_context()
    candidates = [run_one({"task": "A", "model": name, "seed": cfg["seeds"][0], "lr": lr}, cfg) for lr in cfg["learning_rates"]]
    selected = min(candidates, key=lambda item: item["best_val_RMSE"])
    dump(OUT / f"selection_{name}.json", {"model": name, "lr": selected["lr"], "selection_seed": cfg["seeds"][0],
          "selection_metric": "validation_RMSE", "candidates": candidates, "fingerprint": signature()})
    for seed in cfg["seeds"][1:]:
        run_one({"task": "A", "model": name, "seed": seed, "lr": selected["lr"]}, cfg)
    dump(OUT / f"fit_complete_{name}.json", {"completed_at": base.now(), "model": name, "lr": selected["lr"],
          "fingerprint": signature(), "jobs": 7, "selected_seeds": cfg["seeds"], "test_used_for_selection": False})


def fit():
    processes = []
    for name in ("TimeXer", VARIANT):
        handle = (OUT / f"fit_{name}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "--stage", "branch", "--model", name],
                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((name, proc, handle))
    while any(proc.poll() is None for _, proc, _ in processes):
        failed = [name for name, proc, _ in processes if proc.poll() not in (None, 0)]
        dump(OUT / "progress.json", {"updated_at": base.now(), "phase": "experiment3_training",
              "branches": [{"model": name, "pid": proc.pid, "exit_code": proc.poll()} for name, proc, _ in processes],
              "completed_jobs": len(list((OUT / "experiment3").rglob("result.json"))),
              "total_jobs": 14, "failed": failed})
        time.sleep(10)
    for _, _, handle in processes:
        handle.close()
    if any(proc.returncode for _, proc, _ in processes):
        raise RuntimeError("本地训练分支失败，见独立日志；原实验保持不变")


def final_evaluate():
    cfg, audit = setup_context()
    receipts = {name: base.read(OUT / f"fit_complete_{name}.json") for name in ("TimeXer", VARIANT)}
    assert all(item["fingerprint"] == signature() for item in receipts.values())
    plan = base.read(OUT / "plan.json")
    thresholds = plan["thresholds_m"]
    packs = {split: base.load("A", split) for split in ("train", "val", "test")}
    mean, std = base.scale("A")
    device = base.setup(cfg, 42)
    tensors = {split: torch.tensor(pack["x"], device=device) for split, pack in packs.items()}
    rows = []
    for split, pack in packs.items():
        rows += score_rows(pack["persistence"], pack, "Persistence", None, split, thresholds)
    for name in ("TimeXer", VARIANT):
        for seed in cfg["seeds"]:
            job = {"task": "A", "model": name, "lr": receipts[name]["lr"], "seed": seed}
            dest = local_dir(job)
            result = base.read(dest / "result.json")
            assert base.sha256(dest / "best.pt") == result["checkpoint_sha256"]
            base.setup(cfg, seed)
            model = experiment_build(name, audit["tasks"]["A"], cfg, device)
            model.load_state_dict(torch.load(dest / "best.pt", map_location=device, weights_only=True))
            for split, pack in packs.items():
                pred = base.predict(model, tensors[split], cfg["batch_size"]) * std + mean
                rows += score_rows(pred, pack, name, seed, split, thresholds, parameters=result["parameters"],
                                   training_seconds=result["training_seconds"], best_epoch=result["best_epoch"])
                pd.DataFrame({"time": pack["time"], "target_index": pack["target_index"], "observed": pack["y"],
                              "predicted": pred, "persistence": pack["persistence"]}).to_csv(dest / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
            del model
            print(f"EXP3 EVAL {name} seed={seed}", flush=True)
    save_csv(OUT / "experiment3_metrics.csv", rows)
    summary = pd.DataFrame(rows).groupby(["model", "split", "group"], sort=False)[["MAE", "MAPE", "RMSE", "skill_vs_persistence", "bias_m"]].agg(["mean", "std", "count"])
    summary.to_csv(OUT / "experiment3_summary.csv", encoding="utf-8-sig")
    dump(OUT / "completed.json", {"completed_at": base.now(), "status": "completed", "signatures": signatures(),
         "environment": environment(), "training_jobs": 14, "formal_seed_results": 10,
         "baseline_fingerprint_unchanged": base.fingerprint() == ORIGINAL_RECEIPT["fingerprint"],
         "selected_learning_rates": {name: receipt["lr"] for name, receipt in receipts.items()},
         "test_used_for_selection": False, "test_interpretation": "exploratory; previously inspected test"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["diagnose", "fit", "branch", "worker", "evaluate", "all"], default="all")
    parser.add_argument("--job")
    parser.add_argument("--model")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("diagnose", "all"):
        diagnose()
    if args.stage in ("fit", "all"):
        fit()
    if args.stage == "branch":
        branch(args.model)
    if args.stage == "worker":
        worker(json.loads(args.job))
    if args.stage in ("evaluate", "all"):
        final_evaluate()


if __name__ == "__main__":
    main()
