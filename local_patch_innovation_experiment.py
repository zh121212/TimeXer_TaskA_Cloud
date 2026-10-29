"""最近分块细节的单一结构候选及幅值/空标记消融；旧主干与实验均不改。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

import local_weight_decay_diagnosis as prior
import numpy as np
import pandas as pd
import torch

ROOT,base,previous=prior.ROOT,prior.base,prior.previous
OUT=ROOT/"03_实验结果/10_细节幅值结构实验"
MODES={"TimeXer_LastAnchor_RecentRaw":"raw","TimeXer_LastAnchor_RecentNorm":"norm","TimeXer_LastAnchor_RecentZero":"zero"}


class RecentDetail(previous.LastAnchor):
    """复用同一个倒置嵌入；仅增加最近分块的13个外生细节标记。"""
    def __init__(self,metadata,cfg,device,mode):
        super().__init__(metadata,cfg,device)
        self.detail_mode=mode
        self.detail_patch_len=3

    def detail(self,x):
        physical=x[:,:,self.physical_indices]
        external=physical[:,:,:-1]
        recent=external[:,-self.detail_patch_len:]
        residual=recent-recent.mean(dim=1,keepdim=True)
        if self.detail_mode=="norm":
            residual=residual/torch.sqrt(external.var(dim=1,keepdim=True,unbiased=False)+1e-5)
        if self.detail_mode=="zero":
            residual=torch.zeros_like(residual)
        return torch.cat([torch.zeros_like(external[:,:-self.detail_patch_len]),residual],dim=1)

    def forward(self,x):
        self.auxiliary_loss=None
        physical=x[:,:,self.physical_indices]
        calendar=x[:,:,self.calendar_indices]
        marks=torch.cat([calendar,self.detail(x)],dim=-1)
        pred=self.network(physical,marks,None,None)[:,-1,-1:]
        return pred+x[:,-1,-1:]-x[:,:,-1].mean(1,keepdim=True)


def build(name,metadata,cfg,device):
    return RecentDetail(metadata,cfg,device,MODES[name])


def context():
    cfg,audit=prior.context()
    assert base.read(prior.OUT/"verification.json")["status"]=="passed"
    assert base.read(ROOT/"03_实验结果/08_选择性衰减定位/verification.json")["status"]=="passed"
    return {**cfg,"weight_decay":0.0},audit


def parts():
    return {"baseline":base.fingerprint(),"previous_code":base.sha256(previous.__file__),
            "prior_code":base.sha256(prior.__file__),"script":base.sha256(__file__),"weight_decay":0.0,
            "patch_len":3,"source_channels":"13 external physical channels; target excluded","modes":MODES}


def signature():
    return hashlib.sha256(json.dumps(parts(),sort_keys=True).encode()).hexdigest()


def path(job):
    return OUT/"training/01_对比实验/runs"/f"A_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def done(job):
    dest=path(job)
    if not (dest/"result.json").exists():return None
    result=base.read(dest/"result.json")
    assert result["fingerprint"]==signature()
    assert result["checkpoint_sha256"]==base.sha256(dest/"best.pt")
    return result


def interface(cfg,audit):
    device=base.setup(cfg,42)
    x=torch.tensor(base.load("A","train",32)["x"],device=device)
    reference=previous.experiment_build(previous.VARIANT,audit["tasks"]["A"],cfg,device)
    states=reference.state_dict()
    models={name:build(name,audit["tasks"]["A"],cfg,device) for name in MODES}
    for model in models.values():
        model.load_state_dict(states,strict=True);model.eval()
        assert model.state_dict().keys()==states.keys()
        assert sum(p.numel() for p in model.parameters())==102017
    raw=models["TimeXer_LastAnchor_RecentRaw"]
    zero=models["TimeXer_LastAnchor_RecentZero"]
    flat=x.clone();flat[:,:,raw.physical_indices]=flat[:,-1:,raw.physical_indices].expand(-1,12,-1)
    before=x.clone()
    with torch.no_grad():
        detail=raw.detail(x)
        assert detail.shape==(32,12,13)
        torch.testing.assert_close(detail[:,:9],torch.zeros_like(detail[:,:9]),rtol=0,atol=0)
        torch.testing.assert_close(detail[:,-3:].sum(1),torch.zeros_like(detail[:,0]),rtol=0,atol=2e-6)
        torch.testing.assert_close(raw.detail(flat),torch.zeros_like(detail),rtol=0,atol=3e-7)
        torch.testing.assert_close(raw(flat),zero(flat),rtol=0,atol=2e-6)
        capture=[]
        hook=raw.network.ex_embedding.register_forward_pre_hook(lambda module,args:capture.append(args[1].detach().clone()))
        output=raw(x)
        hook.remove()
        assert output.shape==(32,1) and capture[0].shape==(32,12,16)
        torch.testing.assert_close(capture[0][:,:,:3],x[:,:,raw.calendar_indices],rtol=0,atol=0)
        torch.testing.assert_close(capture[0][:,:,3:],detail,rtol=0,atol=0)
        torch.testing.assert_close(x,before,rtol=0,atol=0)
        # 加常量不改变块内细节；变化幅值乘2，细节也恰好乘2（数值容差）。
        doubled=x.clone();doubled[:,:,raw.physical_indices]*=2
        torch.testing.assert_close(raw.detail(doubled),2*detail,rtol=1e-5,atol=1e-6)
    for name,model in models.items():
        model.train();loss=model(x).square().mean();loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum())>0 for g in grads)
    base.write_json(OUT/"interface.json",{"passed":True,"parameters":102017,"all_backbone_parameter_names_shapes_retained":True,
        "calendar_preserved":True,"input_unchanged":True,"recent_patch_zero_sum":True,"amplitude_scaling_verified":True,
        "static_detail_zero":True,"finite_parameter_gradients_all_modes":True,"original_external_tokens":16,"new_external_tokens":29})
    del models,reference,raw,zero,x,flat
    torch.cuda.empty_cache()


def plan():
    cfg,audit=context()
    record={"created_at":base.now(),"signature_parts":parts(),"config":cfg,"environment":previous.environment(),
        "hypothesis":"recent external within-patch fluctuations contain residual predictive information; retaining their global-training-standardized amplitude may avoid amplifying weak fluctuations",
        "single_change":"append 13 recent-patch zero-mean detail tokens to existing shared inverted embedding/cross-attention; no new trainable parameters",
        "controls":{"raw":"global-training-standardized detail, not window-normalized","norm":"same detail divided by existing window stdev","zero":"identical extra token count, all-zero detail"},
        "budget":"each condition: 3 learning rates at seed42 + four other seeds; total21 new jobs; existing TimeXer/LastAnchor wd0 controls reused",
        "primary":"five-seed mean validation RMSE; choose LR by seed42 validation RMSE as before",
        "retain_balanced_module_if":"raw improves mean validation RMSE, MAE, stable RMSE vs LastAnchor wd0; fast RMSE does not worsen; raw beats zero and norm controls on mean RMSE; at least4/5 paired overall RMSE wins vs baseline",
        "interpretation":"exploratory: prior validation residual probes used to form hypothesis; not preregistered independent confirmation; novelty not presumed",
        "test_evaluated":False,"protected_files":prior.protected_files()}
    file=OUT/"plan.json"
    if file.exists():assert base.read(file)["signature_parts"]==parts()
    else:base.write_json(file,record)
    interface(cfg,audit)


def worker(job):
    cfg,audit=context()
    base.RESULTS=OUT/"training"
    base.RUNTIME=OUT/"workers"/f"{job['model']}_s{job['seed']}_lr{job['lr']:g}"
    base.build=build
    base.train_job(cfg,audit,job,signature())


def pool(jobs,cfg):
    pending=[j for j in jobs if done(j) is None];running=[];failed=[]
    while pending or running:
        while pending and len(running)<2 and not failed:
            job=pending.pop(0);dest=path(job)
            if (dest/"epochs.csv").exists() or (dest/"failure.json").exists():raise RuntimeError(f"不覆盖未完成训练：{dest}")
            dest.mkdir(parents=True,exist_ok=True);handle=(dest/"console.log").open("w",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-X","utf8","-u",str(Path(__file__)),"--stage","worker","--job",json.dumps(job)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            running.append((job,proc,handle,time.monotonic()));print(f"TRAIN pid={proc.pid} {job}",flush=True)
        kept=[]
        for job,proc,handle,started in running:
            if proc.poll() is None and time.monotonic()-started>cfg["job_timeout_seconds"]:
                proc.kill();proc.wait();base.write_json(path(job)/"failure.json",{"reason":"hard_timeout","job":job})
            code=proc.poll()
            if code is None:kept.append((job,proc,handle,started))
            else:
                handle.close()
                if code or done(job) is None:failed.append(job)
        running=kept
        if failed:pending=[]
        base.write_json(OUT/"progress.json",{"phase":"structure_training","updated_at":base.now(),"completed_jobs":len(list((OUT/"training").rglob("result.json"))),
            "total_jobs":21,"running":[{"job":job,"pid":proc.pid} for job,proc,_,_ in running],"failed":failed})
        if running:time.sleep(5)
    if failed:raise RuntimeError(f"训练失败，不自动重试：{failed}")


def fit():
    cfg,_=context()
    pool([{"task":"A","model":name,"seed":42,"lr":lr} for name in MODES for lr in cfg["learning_rates"]],cfg)
    for name in MODES:
        candidates=[done({"task":"A","model":name,"seed":42,"lr":lr}) for lr in cfg["learning_rates"]]
        selected=min(candidates,key=lambda row:row["best_val_RMSE"])
        base.write_json(OUT/f"selection_{name}.json",{"model":name,"lr":selected["lr"],"candidates":candidates,"fingerprint":signature()})
    pool([{"task":"A","model":name,"seed":seed,"lr":base.read(OUT/f"selection_{name}.json")["lr"]} for name in MODES for seed in cfg["seeds"][1:]],cfg)
    base.write_json(OUT/"fit_complete.json",{"completed_at":base.now(),"fingerprint":signature(),"jobs":21,"test_evaluated":False})


def evaluate():
    cfg,audit=context()
    assert base.read(OUT/"fit_complete.json")["fingerprint"]==signature()
    thresholds=base.read(previous.OUT/"plan.json")["thresholds_m"]
    packs={s:base.load("A",s) for s in ["train","val"]};mean,std=base.scale("A")
    device=base.setup(cfg,42);tensors={s:torch.tensor(p["x"],device=device) for s,p in packs.items()}
    rows=[];ablations=[];timing=[]
    for name,mode in MODES.items():
        lr=base.read(OUT/f"selection_{name}.json")["lr"]
        for seed in cfg["seeds"]:
            job={"task":"A","model":name,"seed":seed,"lr":lr};result=done(job)
            base.setup(cfg,seed);model=build(name,audit["tasks"]["A"],cfg,device)
            model.load_state_dict(torch.load(path(job)/"best.pt",map_location=device,weights_only=True))
            for split,pack in packs.items():
                pred=base.predict(model,tensors[split],cfg["batch_size"])*std+mean
                rows+=previous.score_rows(pred,pack,name,seed,split,thresholds,mode=mode,lr=lr,best_epoch=result["best_epoch"],parameters=result["parameters"])
                dest=OUT/"predictions"/f"{name}_s{seed}";dest.mkdir(parents=True,exist_ok=True)
                pd.DataFrame({"time":pack["time"],"target_index":pack["target_index"],"observed":pack["y"],"predicted":pred,"persistence":pack["persistence"]}).to_csv(dest/f"{split}.csv",index=False,encoding="utf-8-sig")
            clean=pred
            if mode=="raw":
                for ablated in ["zero","norm"]:
                    model.detail_mode=ablated
                    altered=base.predict(model,tensors['val'],cfg['batch_size'])*std+mean
                    for item in previous.score_rows(altered,packs['val'],name,seed,'val',thresholds):
                        item.update({'ablation':ablated,'prediction_change_RMS':float(np.sqrt(np.mean((altered-clean)**2)))})
                        ablations.append(item)
                model.detail_mode=mode
            if seed==42:
                base.predict(model,tensors['val'],cfg['batch_size'])
                for repeat in range(5):
                    torch.cuda.synchronize();started=time.perf_counter();base.predict(model,tensors['val'],cfg['batch_size']);torch.cuda.synchronize()
                    timing.append({'model':name,'repeat':repeat,'samples':len(packs['val']['y']),'seconds':time.perf_counter()-started})
            previous.save_csv(OUT/'metrics.csv',rows);previous.save_csv(OUT/'inference_ablations.csv',ablations);previous.save_csv(OUT/'timing.csv',timing)
            print(f'EVAL {name} seed={seed}',flush=True);del model
    for name in ['TimeXer',previous.VARIANT]:
        base.setup(cfg,42);model=previous.experiment_build(name,audit['tasks']['A'],cfg,device)
        lr=base.read(prior.selection_path(name,0))['lr'];job={'task':'A','model':name,'seed':42,'lr':lr}
        model.load_state_dict(torch.load(prior.run_path(job,0)/'best.pt',map_location=device,weights_only=True))
        base.predict(model,tensors['val'],cfg['batch_size'])
        for repeat in range(5):
            torch.cuda.synchronize();started=time.perf_counter();base.predict(model,tensors['val'],cfg['batch_size']);torch.cuda.synchronize()
            timing.append({'model':name,'repeat':repeat,'samples':len(packs['val']['y']),'seconds':time.perf_counter()-started})
        del model
    previous.save_csv(OUT/'timing.csv',timing)
    assert prior.protected_files()==base.read(OUT/'plan.json')['protected_files']
    base.write_json(OUT/'completed.json',{'status':'completed','completed_at':base.now(),'signature_parts':parts(),'jobs':21,'formal_seed_results':15,'parameters':102017,
        'test_evaluated':False,'original_files_unchanged':True,'selected_learning_rates':{name:base.read(OUT/f'selection_{name}.json')['lr'] for name in MODES}})
    base.write_json(OUT/'progress.json',{'phase':'completed','completed_jobs':21,'total_jobs':21})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','worker','evaluate'],default='all');p.add_argument('--job');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='worker':worker(json.loads(args.job));return
    if args.stage=='all':plan();fit()
    evaluate()


if __name__=='__main__':main()
