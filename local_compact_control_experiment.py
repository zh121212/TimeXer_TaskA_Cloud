"""三控制标记精简、同数空标记和线性补偿头的独立公平对照。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

import local_patch_innovation_experiment as parent
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT,base,previous=parent.ROOT,parent.base,parent.previous
OUT=ROOT/'03_实验结果/11_精简与线性对照'
MODES={'TimeXer_LastAnchor_Control3Raw':'control3','TimeXer_LastAnchor_Control3Zero':'zero3',
       'TimeXer_LastAnchor_LinearBypass':'linear'}


class Control3Detail(parent.RecentDetail):
    def __init__(self,metadata,cfg,device,mode):
        super().__init__(metadata,cfg,device,'raw' if mode=='control3' else 'zero')
        prefix=metadata['target_station']+':'
        self.control_indices=[i for i,name in enumerate(metadata['features']) if name.startswith(prefix)
                              and (name.split(':',1)[1]=='瞬时流量' or name.endswith('实际开度'))]
        assert len(self.control_indices)==3

    def detail(self,x):
        controls=x[:,:,self.control_indices]
        recent=controls[:,-self.detail_patch_len:]
        residual=recent-recent.mean(1,keepdim=True)
        if self.detail_mode=='zero':residual=torch.zeros_like(residual)
        return torch.cat([torch.zeros_like(controls[:,:-self.detail_patch_len]),residual],dim=1)


class LinearBypass(previous.LastAnchor):
    def __init__(self,metadata,cfg,device):
        super().__init__(metadata,cfg,device)
        # 不改变骨干初始化后的训练随机流；补偿头初始为零。
        with torch.random.fork_rng(devices=[]):
            self.linear_residual=nn.Linear(cfg['seq_len']*metadata['total_channels'],1)
            nn.init.zeros_(self.linear_residual.weight);nn.init.zeros_(self.linear_residual.bias)
        self.to(device)

    def forward(self,x):
        original=super().forward(x)
        centered=x.clone()
        centered[:,:,self.physical_indices]=x[:,:,self.physical_indices]-x[:,-1:,self.physical_indices]
        return original+self.linear_residual(centered.flatten(1))


def build(name,metadata,cfg,device):
    mode=MODES[name]
    return LinearBypass(metadata,cfg,device) if mode=='linear' else Control3Detail(metadata,cfg,device,mode)


def context():
    cfg,audit=parent.context()
    assert base.read(parent.OUT/'verification.json')['status']=='passed'
    assert base.read(parent.OUT/'completed.json')['signature_parts']==parent.parts()
    return cfg,audit


def parts():
    return {'baseline':base.fingerprint(),'previous':base.sha256(previous.__file__),'prior':base.sha256(parent.prior.__file__),
            'parent':base.sha256(parent.__file__),'script':base.sha256(__file__),'weight_decay':0.,'modes':MODES,
            'linear_features':'physical history minus its last observation; calendar unchanged; all12x17 values'}


def signature():return hashlib.sha256(json.dumps(parts(),sort_keys=True).encode()).hexdigest()


def path(job):return OUT/'training/01_对比实验/runs'/f"A_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def done(job):
    dest=path(job)
    if not (dest/'result.json').exists():return None
    result=base.read(dest/'result.json')
    assert result['fingerprint']==signature() and result['checkpoint_sha256']==base.sha256(dest/'best.pt')
    return result


def interface(cfg,audit):
    device=base.setup(cfg,42);meta=audit['tasks']['A'];x=torch.tensor(base.load('A','train',32)['x'],device=device)
    reference=previous.experiment_build(previous.VARIANT,meta,cfg,device);reference.eval()
    original_keys=set(reference.state_dict());records=[]
    for name,mode in MODES.items():
        model=build(name,meta,cfg,device);model.eval()
        missing,unexpected=model.load_state_dict(reference.state_dict(),strict=False)
        assert not unexpected
        assert set(missing)==({'linear_residual.weight','linear_residual.bias'} if mode=='linear' else set())
        assert original_keys.issubset(model.state_dict().keys())
        before=x.clone()
        with torch.no_grad():
            output=model(x);assert output.shape==(32,1)
            torch.testing.assert_close(x,before,rtol=0,atol=0)
            if mode=='linear':torch.testing.assert_close(output,reference(x),rtol=0,atol=0)
            else:
                assert model.control_indices==[13,14,15]
                detail=model.detail(x);assert detail.shape==(32,12,3)
                torch.testing.assert_close(detail[:,:9],torch.zeros_like(detail[:,:9]),rtol=0,atol=0)
                torch.testing.assert_close(detail[:,-3:].sum(1),torch.zeros_like(detail[:,0]),rtol=0,atol=2e-6)
                seen=[];hook=model.network.ex_embedding.register_forward_pre_hook(lambda module,args:seen.append(args[1].detach().clone()))
                model(x);hook.remove();assert seen[0].shape==(32,12,6)
                torch.testing.assert_close(seen[0][:,:,:3],x[:,:,model.calendar_indices],rtol=0,atol=0)
        model.train();model(x).square().mean().backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum())>0 for g in grads)
        records.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),'backbone_retained':True,
                        'input_unchanged':True,'finite_gradients':True,'linear_initially_identical_to_baseline':mode=='linear'})
        del model
    base.write_json(OUT/'interface.json',{'passed':True,'records':records,'control_global_indices':[13,14,15],
        'new_source_tokens':19,'previous_raw13_source_tokens':29,'baseline_source_tokens':16})
    del reference,x;torch.cuda.empty_cache()


def plan():
    cfg,audit=context()
    record={'created_at':base.now(),'signature_parts':parts(),'config':cfg,'environment':previous.environment(),
        'hypothesis':'only local flow and two gate-detail tokens are needed; compare against a zero-token control and a conventional linear residual head',
        'new_jobs':21,'existing_controls':['TimeXer wd0','LastAnchor wd0','RecentRaw13 wd0'],
        'unchanged':['original17rawinputs','12stepwindow','30minutehorizon','calendar','TimeXer backbone','training loss','LR search budget','five seeds'],
        'eligibility':'positive variants must improve mean overall RMSE, MAE and stable RMSE vs LastAnchor; fast RMSE must not worsen; at least4/5 paired overall RMSE wins; control3 must beat zero3 mean RMSE',
        'primary_selection':'eligible candidates plus previously accepted Raw13, ranked by mean five-seed validation RMSE; exact ties prefer fewer parameters then lower current measured latency; zero3 is a negative control',
        'compaction_success':'Control3 RMSE <= Raw13 and current interleaved benchmark median latency < Raw13; report all state tradeoffs',
        'test_evaluated':False,'protected_files':parent.prior.protected_files(),
        'previous_stage_files':{p.relative_to(ROOT).as_posix():base.sha256(p) for p in parent.OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
    file=OUT/'plan.json'
    if file.exists():assert base.read(file)['signature_parts']==parts()
    else:base.write_json(file,record)
    interface(cfg,audit)


def worker(job):
    cfg,audit=context();base.RESULTS=OUT/'training';base.RUNTIME=OUT/'workers'/f"{job['model']}_s{job['seed']}_lr{job['lr']:g}"
    base.build=build;base.train_job(cfg,audit,job,signature())


def pool(jobs,cfg):
    pending=[j for j in jobs if done(j) is None];running=[];failed=[]
    while pending or running:
        while pending and len(running)<2 and not failed:
            job=pending.pop(0);dest=path(job)
            if (dest/'epochs.csv').exists() or (dest/'failure.json').exists():raise RuntimeError(f'不覆盖未完成任务：{dest}')
            dest.mkdir(parents=True,exist_ok=True);handle=(dest/'console.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-X','utf8','-u',str(Path(__file__)),'--stage','worker','--job',json.dumps(job)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            running.append((job,proc,handle,time.monotonic()));print(f'TRAIN pid={proc.pid} {job}',flush=True)
        kept=[]
        for job,proc,handle,started in running:
            if proc.poll() is None and time.monotonic()-started>cfg['job_timeout_seconds']:
                proc.kill();proc.wait();base.write_json(path(job)/'failure.json',{'reason':'hard_timeout','job':job})
            code=proc.poll()
            if code is None:kept.append((job,proc,handle,started))
            else:
                handle.close()
                if code or done(job) is None:failed.append(job)
        running=kept
        if failed:pending=[]
        base.write_json(OUT/'progress.json',{'phase':'training','updated_at':base.now(),'completed_jobs':len(list((OUT/'training').rglob('result.json'))),
            'total_jobs':21,'running':[{'job':job,'pid':proc.pid} for job,proc,_,_ in running],'failed':failed})
        if running:time.sleep(5)
    if failed:raise RuntimeError(f'训练失败，不自动重试：{failed}')


def fit():
    cfg,_=context()
    pool([{'task':'A','model':name,'seed':42,'lr':lr} for name in MODES for lr in cfg['learning_rates']],cfg)
    for name in MODES:
        candidates=[done({'task':'A','model':name,'seed':42,'lr':lr}) for lr in cfg['learning_rates']]
        selected=min(candidates,key=lambda row:row['best_val_RMSE'])
        base.write_json(OUT/f'selection_{name}.json',{'model':name,'lr':selected['lr'],'candidates':candidates,'fingerprint':signature()})
    pool([{'task':'A','model':name,'seed':seed,'lr':base.read(OUT/f'selection_{name}.json')['lr']} for name in MODES for seed in cfg['seeds'][1:]],cfg)
    base.write_json(OUT/'fit_complete.json',{'completed_at':base.now(),'fingerprint':signature(),'jobs':21,'test_evaluated':False})


def evaluate():
    cfg,audit=context();assert base.read(OUT/'fit_complete.json')['fingerprint']==signature()
    packs={s:base.load('A',s) for s in ['train','val']};thresholds=base.read(previous.OUT/'plan.json')['thresholds_m'];mean,std=base.scale('A')
    device=base.setup(cfg,42);tensors={s:torch.tensor(p['x'],device=device) for s,p in packs.items()};rows=[];ablations=[]
    for name,mode in MODES.items():
        lr=base.read(OUT/f'selection_{name}.json')['lr']
        for seed in cfg['seeds']:
            base.setup(cfg,seed);job={'task':'A','model':name,'seed':seed,'lr':lr};result=done(job)
            model=build(name,audit['tasks']['A'],cfg,device);model.load_state_dict(torch.load(path(job)/'best.pt',map_location=device,weights_only=True))
            for split,pack in packs.items():
                pred=base.predict(model,tensors[split],cfg['batch_size'])*std+mean
                rows+=previous.score_rows(pred,pack,name,seed,split,thresholds,mode=mode,lr=lr,parameters=result['parameters'])
                dest=OUT/'predictions'/f'{name}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
                pd.DataFrame({'time':pack['time'],'target_index':pack['target_index'],'observed':pack['y'],'predicted':pred,'persistence':pack['persistence']}).to_csv(dest/f'{split}.csv',index=False,encoding='utf-8-sig')
            clean=pred
            if mode=='control3':
                model.detail_mode='zero'
                disabled=base.predict(model,tensors['val'],cfg['batch_size'])*std+mean
                for row in previous.score_rows(disabled,packs['val'],name,seed,'val',thresholds):row.update({'ablation':'disable_control_detail'});ablations.append(row)
            if mode=='linear':
                with torch.no_grad():model.linear_residual.weight.zero_();model.linear_residual.bias.zero_()
                disabled=base.predict(model,tensors['val'],cfg['batch_size'])*std+mean
                for row in previous.score_rows(disabled,packs['val'],name,seed,'val',thresholds):row.update({'ablation':'disable_linear_residual'});ablations.append(row)
            previous.save_csv(OUT/'metrics.csv',rows);previous.save_csv(OUT/'inference_ablations.csv',ablations)
            print(f'EVAL {name} seed={seed}',flush=True);del model
    # 同时段、同GPU、轮换顺序，避免沿用不同实验时刻的耗时比较。
    models={}
    for name in MODES:
        lr=base.read(OUT/f'selection_{name}.json')['lr'];job={'task':'A','model':name,'seed':42,'lr':lr};model=build(name,audit['tasks']['A'],cfg,device)
        model.load_state_dict(torch.load(path(job)/'best.pt',map_location=device,weights_only=True));models[name]=model
    old_name='TimeXer_LastAnchor_RecentRaw';lr=base.read(parent.OUT/f'selection_{old_name}.json')['lr'];job={'task':'A','model':old_name,'seed':42,'lr':lr}
    model=parent.build(old_name,audit['tasks']['A'],cfg,device);model.load_state_dict(torch.load(parent.path(job)/'best.pt',map_location=device,weights_only=True));models[old_name]=model
    old_name=previous.VARIANT;model=previous.experiment_build(old_name,audit['tasks']['A'],cfg,device);job={'task':'A','model':old_name,'seed':42,'lr':.001}
    model.load_state_dict(torch.load(parent.prior.run_path(job,0)/'best.pt',map_location=device,weights_only=True));models[old_name]=model
    for model in models.values():base.predict(model,tensors['val'],cfg['batch_size'])
    timings=[];names=list(models)
    for repeat in range(10):
        order=names[repeat%len(names):]+names[:repeat%len(names)]
        for name in order:
            torch.cuda.synchronize();started=time.perf_counter();base.predict(models[name],tensors['val'],cfg['batch_size']);torch.cuda.synchronize()
            timings.append({'model':name,'repeat':repeat,'samples':len(packs['val']['y']),'seconds':time.perf_counter()-started})
    previous.save_csv(OUT/'timing.csv',timings)
    plan=base.read(OUT/'plan.json');assert parent.prior.protected_files()==plan['protected_files']
    for relative,digest in plan['previous_stage_files'].items():assert base.sha256(ROOT/relative)==digest
    base.write_json(OUT/'completed.json',{'status':'completed','completed_at':base.now(),'signature_parts':parts(),'jobs':21,'formal_seed_results':15,
        'test_evaluated':False,'old_files_unchanged':True,'selected_learning_rates':{name:base.read(OUT/f'selection_{name}.json')['lr'] for name in MODES}})
    base.write_json(OUT/'progress.json',{'phase':'completed','completed_jobs':21,'total_jobs':21})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','worker','evaluate'],default='all');p.add_argument('--job');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='worker':worker(json.loads(args.job));return
    if args.stage=='all':plan();fit()
    evaluate()


if __name__=='__main__':main()
