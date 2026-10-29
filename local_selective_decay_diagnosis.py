"""定位衰减作用位置：仅外生嵌入和交叉注意力免衰减，模型结构不变。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

import local_weight_decay_diagnosis as comparison
import numpy as np
import pandas as pd
import torch

base, previous, ROOT = comparison.base, comparison.previous, comparison.ROOT
OUT = ROOT / "03_实验结果/08_选择性衰减定位"
NAME = "TimeXer_LastAnchor_PathDecay"
EXEMPT = {"external_embedding", "cross_query_key", "cross_value_output"}


def signature_parts():
    return {"baseline":base.fingerprint(), "previous_script":base.sha256(previous.__file__),
            "comparison_script":base.sha256(comparison.__file__), "script":base.sha256(__file__),
            "stage_one_receipt":base.sha256(comparison.OUT/"completed.json"),
            "exempt_groups":sorted(EXEMPT),"remaining_decay":0.0001}


def signature():
    return hashlib.sha256(json.dumps(signature_parts(),sort_keys=True).encode()).hexdigest()


def context():
    cfg,audit=comparison.context()
    assert base.read(comparison.OUT/"verification.json")["status"]=="passed"
    assert base.read(comparison.OUT/"completed.json")["signatures"]==comparison.sig_parts()
    return cfg,audit


def path(job):
    return OUT/"training/01_对比实验/runs"/f"A_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def complete(job):
    dest=path(job)
    if not (dest/"result.json").exists():
        return None
    result=base.read(dest/"result.json")
    assert result["fingerprint"]==signature()
    assert base.sha256(dest/"best.pt")==result["checkpoint_sha256"]
    return result


def worker(job):
    cfg,audit=context()
    base.RESULTS=OUT/"training"
    base.RUNTIME=OUT/"workers"/f"s{job['seed']}_lr{job['lr']:g}"
    state={}
    def build(name,meta,cfg,device):
        assert name==NAME
        model=previous.experiment_build(previous.VARIANT,meta,cfg,device)
        state["model"]=model
        return model
    original_adam=torch.optim.Adam
    def selective_adam(parameters,lr,weight_decay,**kwargs):
        assert weight_decay==0.0001
        params=list(parameters)
        names={id(p):name for name,p in state["model"].named_parameters()}
        exempt=[p for p in params if comparison.parameter_group(names[id(p)]) in EXEMPT]
        remaining=[p for p in params if comparison.parameter_group(names[id(p)]) not in EXEMPT]
        assert len(exempt)+len(remaining)==len(params) and exempt and remaining
        assert sum(p.numel() for p in params)==102017
        record={"exempt_names":[names[id(p)] for p in exempt], "exempt_decay":0.0,
                "remaining_names":[names[id(p)] for p in remaining],"remaining_decay":weight_decay,
                "exempt_parameters":sum(p.numel() for p in exempt),"remaining_parameters":sum(p.numel() for p in remaining)}
        base.write_json(path(job)/"optimizer_groups.json",record)
        return original_adam([{"params":exempt,"weight_decay":0.0},{"params":remaining,"weight_decay":weight_decay}],lr=lr,weight_decay=weight_decay,**kwargs)
    base.build=build
    torch.optim.Adam=selective_adam
    try:
        base.train_job(cfg,audit,job,signature())
    finally:
        torch.optim.Adam=original_adam


def jobs_in_parallel(jobs,cfg):
    pending=[job for job in jobs if complete(job) is None]
    running=[]
    failed=[]
    while pending or running:
        while pending and len(running)<2 and not failed:
            job=pending.pop(0)
            dest=path(job)
            if (dest/"epochs.csv").exists() or (dest/"failure.json").exists():
                raise RuntimeError(f"不覆盖未完成任务：{dest}")
            dest.mkdir(parents=True,exist_ok=True)
            handle=(dest/"console.log").open("w",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-X","utf8","-u",str(Path(__file__)),"--stage","worker","--job",json.dumps(job)],
                                  cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            running.append((job,proc,handle,time.monotonic()))
            print(f"TRAIN pid={proc.pid} {job}",flush=True)
        retained=[]
        for job,proc,handle,started in running:
            if proc.poll() is None and time.monotonic()-started>cfg["job_timeout_seconds"]:
                proc.kill();proc.wait()
                base.write_json(path(job)/"failure.json",{"reason":"hard_timeout","job":job})
            code=proc.poll()
            if code is None:
                retained.append((job,proc,handle,started))
            else:
                handle.close()
                if code or complete(job) is None:
                    failed.append(job)
        running=retained
        if failed:
            pending=[]
        base.write_json(OUT/"progress.json",{"phase":"selective_decay_training","updated_at":base.now(),
            "completed_jobs":len(list((OUT/"training").rglob("result.json"))),"total_jobs":7,
            "running":[{"job":job,"pid":proc.pid} for job,proc,_,_ in running],"failed":failed})
        if running:
            time.sleep(5)
    if failed:
        raise RuntimeError(f"训练失败，不自动重试：{failed}")


def fit():
    cfg,_=context()
    candidates=[{"task":"A","model":NAME,"seed":42,"lr":lr} for lr in cfg["learning_rates"]]
    jobs_in_parallel(candidates,cfg)
    result=min([complete(job) for job in candidates],key=lambda row:row["best_val_RMSE"])
    base.write_json(OUT/"selection.json",{"model":NAME,"lr":result["lr"],"seed":42,"metric":"validation_RMSE",
        "candidates":[complete(job) for job in candidates],"fingerprint":signature()})
    jobs_in_parallel([{"task":"A","model":NAME,"seed":seed,"lr":result["lr"]} for seed in cfg["seeds"][1:]],cfg)
    base.write_json(OUT/"fit_complete.json",{"completed_at":base.now(),"fingerprint":signature(),"jobs":7,"lr":result["lr"],"test_evaluated":False})


def evaluate():
    cfg,audit=context()
    assert base.read(OUT/"fit_complete.json")["fingerprint"]==signature()
    selected=base.read(OUT/"selection.json")["lr"]
    thresholds=base.read(previous.OUT/"plan.json")["thresholds_m"]
    packs={s:base.load("A",s) for s in ["train","val"]}
    device=base.setup(cfg,42)
    tensors={s:torch.tensor(p["x"],device=device) for s,p in packs.items()}
    mean,std=base.scale("A")
    rows=[];permutations=[];norms=[];attention=[]
    groups={"upstream_physical":list(range(8)),"local_physical":list(range(11,17)),"calendar":[8,9,10]}
    for seed in cfg["seeds"]:
        job={"task":"A","model":NAME,"seed":seed,"lr":selected}
        result=complete(job)
        base.setup(cfg,seed)
        model=previous.experiment_build(previous.VARIANT,audit["tasks"]["A"],cfg,device)
        initial={k:v.detach().clone() for k,v in model.named_parameters()}
        model.load_state_dict(torch.load(path(job)/"best.pt",map_location=device,weights_only=True))
        norms+=comparison.norms(model,initial,NAME,seed,None,selected)
        for split,pack in packs.items():
            if split=="val":
                pred,stats=comparison.attention_statistics(model,tensors[split],cfg["batch_size"])
                attention += [{"model":NAME,"seed":seed,**item} for item in stats]
            else:
                pred=base.predict(model,tensors[split],cfg["batch_size"])
            pred=pred*std+mean
            rows+=previous.score_rows(pred,pack,NAME,seed,split,thresholds,condition="selective_decay",lr=selected,best_epoch=result["best_epoch"])
            dest=OUT/"predictions"/f"s{seed}";dest.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({"time":pack["time"],"target_index":pack["target_index"],"observed":pack["y"],"predicted":pred,
                          "persistence":pack["persistence"]}).to_csv(dest/f"{split}.csv",index=False,encoding="utf-8-sig")
        clean=pred
        score=base.metrics(clean,packs["val"]["y"],packs["val"]["persistence"])
        for group,indices in groups.items():
            for repeat in range(10):
                order=torch.tensor(np.random.default_rng(8100+repeat).permutation(len(clean)),device=device)
                x=tensors["val"].clone();x[:,:,indices]=tensors["val"][order][:,:,indices]
                altered=base.predict(model,x,cfg["batch_size"])*std+mean
                changed=base.metrics(altered,packs["val"]["y"],packs["val"]["persistence"])
                permutations.append({"model":NAME,"seed":seed,"group":group,"repeat":repeat,"clean_RMSE":score["RMSE"],
                    "delta_RMSE":changed["RMSE"]-score["RMSE"],"prediction_shift_RMS_m":float(np.sqrt(np.mean((altered-clean)**2))),**changed})
        previous.save_csv(OUT/"metrics.csv",rows);previous.save_csv(OUT/"permutations.csv",permutations)
        previous.save_csv(OUT/"parameter_norms.csv",norms);previous.save_csv(OUT/"attention.csv",attention)
        print(f"EVAL SELECTIVE seed={seed}",flush=True)
        del model,initial
    assert comparison.protected_files()==base.read(comparison.OUT/"plan.json")["protected_files"]
    base.write_json(OUT/"completed.json",{"status":"completed","completed_at":base.now(),"signature_parts":signature_parts(),
        "jobs":7,"parameters":102017,"test_evaluated":False,"architecture_changed":False,"selected_lr":selected})
    base.write_json(OUT/"progress.json",{"phase":"completed","completed_jobs":7,"total_jobs":7})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","worker","evaluate"],default="all")
    parser.add_argument("--job")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=="worker":
        worker(json.loads(args.job));return
    if args.stage=="all":
        cfg,_=context()
        plan={"created_at":base.now(),"signature_parts":signature_parts(),"base_config":cfg,
              "only_change":"external_embedding and all cross_attention parameters exempt from decay; remaining parameters keep 1e-4",
              "hypothesis":"separate external-path shrinkage from effects of removing regularization on the full target backbone",
              "budget":"3 learning rates at seed42 then four other seeds, seven jobs; same data/model/objective",
              "test_evaluated":False,"architecture_changed":False}
        if (OUT/"plan.json").exists():
            assert base.read(OUT/"plan.json")["signature_parts"]==signature_parts()
        else:
            base.write_json(OUT/"plan.json",plan)
        fit()
    evaluate()


if __name__=="__main__":
    main()
