"""独立权重衰减原因验证：仅在内存中覆盖weight_decay，不改原公共协议。"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time

import local_three_experiments as previous
import numpy as np
import pandas as pd
import torch

ROOT, base = previous.ROOT, previous.base
OUT = ROOT / "03_实验结果/07_权重衰减原因验证"
MODELS = ("TimeXer", previous.VARIANT)
OLD = previous.OUT
DECAYS = (0.0001, 0.0)


def dump(path, value):
    base.write_json(path, value)


def sig_parts():
    return {"baseline": base.fingerprint(), "previous_script_sha256": base.sha256(previous.__file__),
            "current_script_sha256": base.sha256(__file__), "override": {"weight_decay": 0.0}}


def signature():
    return hashlib.sha256(json.dumps(sig_parts(), sort_keys=True).encode()).hexdigest()


def context():
    cfg, audit = previous.setup_context()
    completed = base.read(OLD / "completed.json")
    assert base.sha256(previous.__file__) == completed["signatures"]["experiment_script_sha256"]
    assert base.fingerprint() == completed["signatures"]["baseline"]
    assert cfg["weight_decay"] == 0.0001
    return cfg, audit


def run_path(job, decay):
    root = OLD / "experiment3" if decay == 0.0001 else OUT / "training_wd0"
    return root / "01_对比实验/runs" / f"A_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def selection_path(name, decay):
    return (OLD if decay == 0.0001 else OUT) / f"selection_{name}.json"


def protected_files():
    roots = [ROOT / "03_实验结果/01_对比实验", OLD]
    paths = [ROOT / "local_three_experiments.py", ROOT / "04_实验配置/protocol.json"]
    for folder in roots:
        paths += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return {p.relative_to(ROOT).as_posix(): base.sha256(p) for p in paths}


def plan():
    cfg, _ = context()
    value = {"created_at": base.now(), "status": "planned_before_new_training", "signatures": sig_parts(),
             "environment": previous.environment(), "base_config": cfg, "only_training_override": {"weight_decay": 0.0},
             "hypothesis": "coupled Adam weight decay contributes to near-zero cross-attention projections",
             "models": list(MODELS), "decays": list(DECAYS), "new_jobs": 14, "reused_control_jobs": 14,
             "seeds": cfg["seeds"], "learning_rates": cfg["learning_rates"],
             "selection": "seed42 validation RMSE; identical LR search budget; all other seeds use selected LR",
             "evaluation": "train and val only; never load test pack or test predictions in this round",
             "diagnostics": ["physical-scale MAE/RMSE and training-threshold regimes", "10 grouped permutations per feature group/seed/condition",
                             "initial vs trained projection norms", "validation attention deviations from uniform",
                             "seed42 checkpoints at each shared learning rate", "initial data-gradient vs decay-gradient norms over 4 training mini-batches"],
             "decision": "parameter/sensitivity recovery alone is insufficient; report paired validation errors, persistence comparison, regime tradeoffs and seed consistency before any structural change",
             "protected_files": protected_files()}
    path = OUT / "plan.json"
    if path.exists():
        old = base.read(path)
        assert old["signatures"] == value["signatures"], "已有本轮不同版本，不能覆盖"
        assert old["protected_files"] == value["protected_files"], "原有实验文件变化"
    else:
        dump(path, value)


def worker(job):
    cfg, audit = context()
    changed = {**cfg, "weight_decay": 0.0}
    assert {k for k in cfg if cfg[k] != changed[k]} == {"weight_decay"}
    base.RESULTS = OUT / "training_wd0"
    base.RUNTIME = OUT / "workers" / f"{job['model']}_{job['seed']}_{job['lr']:g}"
    base.RUNTIME.mkdir(parents=True, exist_ok=True)
    base.build = previous.experiment_build
    base.train_job(changed, audit, job, signature())


def one(job, cfg):
    dest = run_path(job, 0.0)
    if (dest / "result.json").exists():
        result = base.read(dest / "result.json")
        assert result["fingerprint"] == signature()
        assert result["checkpoint_sha256"] == base.sha256(dest / "best.pt")
        return result
    if (dest / "epochs.csv").exists() or (dest / "failure.json").exists():
        raise RuntimeError(f"发现未完成训练，不覆盖：{dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "console.log").open("w", encoding="utf-8") as handle:
        child = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "--stage", "worker", "--job", json.dumps(job)],
                                 cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        print(f"TRAIN pid={child.pid} {job}", flush=True)
        try:
            code = child.wait(timeout=cfg["job_timeout_seconds"])
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            dump(dest / "failure.json", {"reason": "hard_timeout", "job": job})
            raise RuntimeError(f"训练超时：{job}")
        if code:
            dump(dest / "failure.json", {"reason": "worker_exit", "code": code, "job": job})
            raise RuntimeError(f"训练失败：{dest / 'console.log'}")
    return base.read(dest / "result.json")


def branch(name):
    cfg, _ = context()
    candidates = [one({"task": "A", "model": name, "seed": cfg["seeds"][0], "lr": lr}, cfg) for lr in cfg["learning_rates"]]
    selected = min(candidates, key=lambda item: item["best_val_RMSE"])
    dump(selection_path(name, 0.0), {"model": name, "lr": selected["lr"], "decay": 0.0, "candidates": candidates,
                                  "selection_seed": cfg["seeds"][0], "metric": "validation_RMSE", "fingerprint": signature()})
    for seed in cfg["seeds"][1:]:
        one({"task": "A", "model": name, "seed": seed, "lr": selected["lr"]}, cfg)
    dump(OUT / f"fit_complete_{name}.json", {"model": name, "completed_at": base.now(), "fingerprint": signature(),
                                          "lr": selected["lr"], "jobs": 7, "test_evaluated": False})


def fit():
    running = []
    for name in MODELS:
        handle = (OUT / f"fit_{name}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "--stage", "branch", "--model", name],
                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        running.append((name, proc, handle))
    while any(proc.poll() is None for _, proc, _ in running):
        dump(OUT / "progress.json", {"phase": "training_wd0", "updated_at": base.now(),
             "completed_jobs": len(list((OUT / "training_wd0").rglob("result.json"))), "total_jobs": 14,
             "branches": [{"model": name, "pid": proc.pid, "exit_code": proc.poll()} for name, proc, _ in running]})
        time.sleep(10)
    for _, _, handle in running:
        handle.close()
    if any(proc.returncode for _, proc, _ in running):
        raise RuntimeError("分支训练失败，见独立日志；不自动重试")


def parameter_group(name):
    if ".cross_attention." in name:
        return "cross_query_key" if ("query_projection" in name or "key_projection" in name) else "cross_value_output"
    if ".ex_embedding." in name:
        return "external_embedding"
    if ".en_embedding." in name:
        return "target_embedding"
    if ".self_attention." in name:
        return "self_attention"
    if ".head." in name:
        return "prediction_head"
    return "remaining_parameters"


def gradient_probe():
    cfg, audit = context()
    train = base.load("A", "train")
    mean, std = base.scale("A")
    rows = []
    device = base.setup(cfg, 42)
    adam = torch.optim.Adam([torch.nn.Parameter(torch.ones(1, device=device))], weight_decay=cfg["weight_decay"])
    dump(OUT / "optimizer_semantics.json", {"class": "torch.optim.Adam", "signature": str(inspect.signature(torch.optim.Adam)),
         "decoupled_weight_decay": bool(adam.param_groups[0].get("decoupled_weight_decay", False)),
         "note": "when false, effective gradient contains weight_decay * parameter, not AdamW"})
    del adam
    for name in MODELS:
        for seed in cfg["seeds"]:
            base.setup(cfg, seed)
            model = previous.experiment_build(name, audit["tasks"]["A"], cfg, device)
            model.train()
            rng = np.random.default_rng(9017)
            for batch in range(4):
                ids = rng.choice(len(train["y"]), size=cfg["batch_size"], replace=False)
                x = torch.tensor(train["x"][ids], device=device)
                y = torch.tensor((train["y"][ids]-mean)/std, dtype=torch.float32, device=device)[:, None]
                model.zero_grad(set_to_none=True)
                loss = base.training_loss(model, model(x), y, cfg)
                loss.backward()
                sums = {}
                for key, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    group = parameter_group(key)
                    item = sums.setdefault(group, {"data_sq": 0.0, "decay_sq": 0.0, "dot": 0.0, "parameters": 0})
                    g, decay = p.grad.detach(), p.detach()*cfg["weight_decay"]
                    item["data_sq"] += float(g.square().sum())
                    item["decay_sq"] += float(decay.square().sum())
                    item["dot"] += float((g*decay).sum())
                    item["parameters"] += p.numel()
                for group, item in sums.items():
                    a, b = math.sqrt(item["data_sq"]), math.sqrt(item["decay_sq"])
                    rows.append({"model": name, "seed": seed, "batch": batch, "group": group, "loss": float(loss.detach()),
                                 "data_gradient_norm": a, "decay_gradient_norm": b,
                                 "decay_to_data_ratio": b/a if a else None, "parameters": item["parameters"]})
            del model
            print(f"GRADIENT PROBE {name} seed={seed}", flush=True)
    previous.save_csv(OUT / "initial_gradient_comparison.csv", rows)


def norms(model, initial, name, seed, decay, lr):
    rows = []
    for key, p in model.named_parameters():
        a, b = float(initial[key].norm()), float(p.detach().norm())
        rows.append({"model": name, "seed": seed, "weight_decay": decay, "lr": lr, "parameter": key,
                     "group": parameter_group(key), "initial_norm": a, "trained_norm": b, "ratio": b/a if a else None})
    return rows


def attention_statistics(model, x, batch_size):
    state, handles = {}, []
    for index, layer in enumerate(model.network.encoder.layers):
        attention = layer.cross_attention.inner_attention
        flag = attention.output_attention
        attention.output_attention = True
        def hook(module, args, output, index=index):
            a = output[1].detach()
            uniform = 1/a.shape[-1]
            item = state.setdefault(index, {"n": 0, "abs_sum": 0.0, "max_deviation": 0.0, "entropy_sum": 0.0, "distributions": 0})
            item["n"] += a.numel()
            item["abs_sum"] += float((a-uniform).abs().sum())
            item["max_deviation"] = max(item["max_deviation"], float((a-uniform).abs().max()))
            entropy = -(a*a.clamp_min(1e-30).log()).sum(-1)/math.log(a.shape[-1])
            item["entropy_sum"] += float(entropy.sum())
            item["distributions"] += entropy.numel()
        handles.append((attention, flag, attention.register_forward_hook(hook)))
    try:
        prediction = base.predict(model, x, batch_size)
    finally:
        for attention, flag, handle in handles:
            handle.remove()
            attention.output_attention = flag
    return prediction, [{"layer": i, "mean_abs_deviation_from_uniform": s["abs_sum"]/s["n"],
                          "max_deviation_from_uniform": s["max_deviation"],
                          "normalized_entropy": s["entropy_sum"]/s["distributions"]} for i, s in state.items()]


def evaluate():
    cfg, audit = context()
    for name in MODELS:
        assert base.read(OUT / f"fit_complete_{name}.json")["fingerprint"] == signature()
    thresholds = base.read(OLD / "plan.json")["thresholds_m"]
    packs = {s: base.load("A", s) for s in ("train", "val")}
    mean, std = base.scale("A")
    device = base.setup(cfg, 42)
    tensors = {s: torch.tensor(p["x"], device=device) for s, p in packs.items()}
    feature_groups = {"upstream_physical": list(range(8)), "local_physical": list(range(11,17)), "calendar": [8,9,10]}
    permutations = [torch.tensor(np.random.default_rng(8100+i).permutation(len(packs["val"]["y"])), device=device) for i in range(10)]
    rows, perm_rows, norm_rows, attention_rows, matched = [], [], [], [], []
    for split, pack in packs.items():
        rows += previous.score_rows(pack["persistence"], pack, "Persistence", None, split, thresholds, weight_decay=None)
    for decay in DECAYS:
        for name in MODELS:
            selected = base.read(selection_path(name, decay))
            for seed in cfg["seeds"]:
                job = {"task": "A", "model": name, "seed": seed, "lr": selected["lr"]}
                path = run_path(job, decay)
                result = base.read(path / "result.json")
                assert base.sha256(path / "best.pt") == result["checkpoint_sha256"]
                base.setup(cfg, seed)
                model = previous.experiment_build(name, audit["tasks"]["A"], cfg, device)
                initial = {k: v.detach().clone() for k,v in model.named_parameters()}
                model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True))
                norm_rows += norms(model, initial, name, seed, decay, selected["lr"])
                for split, pack in packs.items():
                    if split == "val":
                        standardized, attention = attention_statistics(model, tensors[split], cfg["batch_size"])
                        attention_rows += [{"model": name, "seed": seed, "weight_decay": decay, **item} for item in attention]
                    else:
                        standardized = base.predict(model, tensors[split], cfg["batch_size"])
                    pred = standardized*std + mean
                    rows += previous.score_rows(pred, pack, name, seed, split, thresholds,
                                                weight_decay=decay, lr=selected["lr"], best_epoch=result["best_epoch"])
                    dest = OUT / "predictions" / f"{name}_wd{decay:g}_s{seed}"
                    dest.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"time": pack["time"], "target_index": pack["target_index"], "observed": pack["y"],
                                  "predicted": pred, "persistence": pack["persistence"]}).to_csv(dest / f"{split}.csv", index=False, encoding="utf-8-sig")
                clean = pred
                clean_rmse = base.metrics(clean, packs["val"]["y"], packs["val"]["persistence"])["RMSE"]
                for group, indices in feature_groups.items():
                    for repeat, permutation in enumerate(permutations):
                        altered = tensors["val"].clone()
                        altered[:, :, indices] = tensors["val"][permutation][:, :, indices]
                        prediction = base.predict(model, altered, cfg["batch_size"])*std+mean
                        score = base.metrics(prediction, packs["val"]["y"], packs["val"]["persistence"])
                        perm_rows.append({"model": name, "seed": seed, "weight_decay": decay, "group": group, "repeat": repeat,
                                          "clean_RMSE": clean_rmse, "delta_RMSE": score["RMSE"]-clean_rmse,
                                          "prediction_shift_RMS_m": float(np.sqrt(np.mean((prediction-clean)**2))), **score})
                print(f"EVAL {name} wd={decay:g} seed={seed}", flush=True)
                previous.save_csv(OUT / "metrics.csv", rows)
                previous.save_csv(OUT / "permutations.csv", perm_rows)
                previous.save_csv(OUT / "parameter_norms.csv", norm_rows)
                previous.save_csv(OUT / "attention.csv", attention_rows)
                del model, initial
            for lr in cfg["learning_rates"]:
                job = {"task": "A", "model": name, "seed": cfg["seeds"][0], "lr": lr}
                result = base.read(run_path(job, decay) / "result.json")
                epochs = pd.read_csv(run_path(job, decay) / "epochs.csv")
                matched.append({"model": name, "weight_decay": decay, "lr": lr, "seed": cfg["seeds"][0],
                                "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                                "best_val_RMSE": result["best_val_RMSE"], "best_val_MAE": float(epochs.loc[epochs.val_RMSE.idxmin(),"val_MAE"]),
                                "first_val_RMSE": float(epochs.iloc[0].val_RMSE)})
    previous.save_csv(OUT / "matched_lr_seed42.csv", matched)
    assert protected_files() == base.read(OUT / "plan.json")["protected_files"]
    dump(OUT / "completed.json", {"status": "completed", "completed_at": base.now(), "signatures": sig_parts(),
         "new_training_jobs": 14, "reused_control_jobs": 14, "evaluated_selected_checkpoints": 20,
         "test_evaluated": False, "original_files_unchanged": True, "parameters": 102017,
         "selected_learning_rates": {f"{name}/wd{decay:g}":base.read(selection_path(name,decay))["lr"] for name in MODELS for decay in DECAYS}})
    dump(OUT / "progress.json", {"phase": "completed", "completed_jobs": 14, "total_jobs": 14, "test_evaluated": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all","plan","gradient","fit","branch","worker","evaluate"], default="all")
    parser.add_argument("--job")
    parser.add_argument("--model")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("plan", "all"):
        plan()
    if args.stage in ("gradient", "all"):
        gradient_probe()
    if args.stage in ("fit", "all"):
        fit()
    if args.stage == "branch":
        branch(args.model)
    if args.stage == "worker":
        worker(json.loads(args.job))
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
