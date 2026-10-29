"""仅训练/验证段：冻结预测的误差分解与历史信息残差探针，不改旧模型。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"03_实验结果/09_结构机制验证"
PACKS=ROOT/"01_数据集/03_划分打包/A"
SEEDS=[42,123,2024,2025,3407]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding="utf-8-sig")


def scores(pred,y,persistence):
    return {"RMSE":float(np.sqrt(np.mean((pred-y)**2))),"MAE":float(np.mean(np.abs(pred-y)))}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    packs={}
    for split in ["train","val"]:
        with np.load(PACKS/f"{split}.npz") as p:
            packs[split]={k:p[k] for k in p.files}
    physical=[i for i in range(17) if i not in [8,9,10]]
    train_delta=np.abs(packs['train']['y']-packs['train']['persistence'])
    thresholds=np.quantile(train_delta,[.5,.9])
    features={}
    for split,p in packs.items():
        x=p["x"].astype(np.float64);phys=x[:,:,physical]
        means=phys.mean(1,keepdims=True);std=np.sqrt(phys.var(1,keepdims=True)+1e-5)
        dynamics=((phys-means)/std).reshape(len(x),-1)
        state=means[:,0,:]
        scales=np.sqrt(phys.var(1))
        calendar=x[:,:,[8,9,10]].reshape(len(x),-1)
        features[split]={"dynamics":dynamics,"absolute_state":state,"absolute_and_scale":np.c_[state,scales],
                         "calendar":calendar,"dynamics_and_state":np.c_[dynamics,state,scales],
                         "all_views":np.c_[dynamics,state,scales,calendar]}
    rows=[];lags=[];probes=[];shrink=[];files=[]
    source_conditions=[("TimeXer_wd0","07_权重衰减原因验证/predictions/TimeXer_wd0_s{seed}"),
        ("LastAnchor_wd0","07_权重衰减原因验证/predictions/TimeXer_LastAnchor_wd0_s{seed}"),
        ("LastAnchor_wd1e4","07_权重衰减原因验证/predictions/TimeXer_LastAnchor_wd0.0001_s{seed}"),
        ("LastAnchor_PathDecay","08_选择性衰减定位/predictions/s{seed}")]
    matrices={}
    for feature in features['train']:
        fit=features['train'][feature];mu=fit.mean(0);sd=fit.std(0);sd=np.where(sd>1e-6,sd,1.)
        z=(fit-mu)/sd
        # 固定超参数，仅用于信息可用性诊断，不当作正式模型或创新收益。
        matrices[feature]=(z,np.linalg.solve(z.T@z+len(z)*1e-3*np.eye(z.shape[1]),z.T),mu,sd)
    for condition,folder in source_conditions:
        for seed in SEEDS:
            prediction={}
            for split,p in packs.items():
                file=ROOT/'03_实验结果'/folder.format(seed=seed)/f'{split}.csv'
                files.append(file)
                frame=pd.read_csv(file)
                np.testing.assert_array_equal(frame.target_index,p['target_index'])
                pred=frame.predicted.to_numpy();prediction[split]=pred
                actual=p['y']-p['persistence'];delta=pred-p['persistence'];error=pred-p['y']
                masks={"all":np.ones(len(pred),bool),"stable":np.abs(actual)<=thresholds[0],
                       "moderate":(np.abs(actual)>thresholds[0])&(np.abs(actual)<=thresholds[1]),"fast":np.abs(actual)>thresholds[1]}
                for group,mask in masks.items():
                    a,d,e=actual[mask],delta[mask],error[mask]
                    row={"condition":condition,"seed":seed,"split":split,"group":group,"n":int(mask.sum()),
                         "bias":float(e.mean()),"MSE":float(np.mean(e**2)),"bias_squared":float(e.mean()**2),
                         "centered_error_variance":float(np.mean((e-e.mean())**2)),
                         "predicted_change_energy":float(np.mean(d**2)),"true_change_energy":float(np.mean(a**2)),
                         "twice_alignment":float(2*np.mean(a*d)),"excess_MSE_over_persistence":float(np.mean(d*d-2*a*d)),
                         "predicted_change_RMS":float(np.sqrt(np.mean(d*d))),"true_change_RMS":float(np.sqrt(np.mean(a*a))),
                         **scores(pred[mask],p['y'][mask],p['persistence'][mask])}
                    assert abs(row['MSE']-row['bias_squared']-row['centered_error_variance'])<1e-12
                    rows.append(row)
                for lag in range(-6,7):
                    if lag>0:a,b=delta[:-lag],actual[lag:]
                    elif lag<0:a,b=delta[-lag:],actual[:lag]
                    else:a,b=delta,actual
                    lags.append({"condition":condition,"seed":seed,"split":split,"lag_steps":lag,
                                 "correlation":float(np.corrcoef(a,b)[0,1])})
            train=packs['train'];val=packs['val']
            residual=train['y']-prediction['train'];intercept=float(residual.mean())
            train_correction=prediction['train']-train['persistence']
            alpha=float(np.clip(np.dot(train_correction,train['y']-train['persistence'])/np.dot(train_correction,train_correction),0,1))
            for split,p in packs.items():
                pred=p['persistence']+alpha*(prediction[split]-p['persistence'])
                shrink.append({"condition":condition,"seed":seed,"split":split,"alpha_fit_train":alpha,
                               "probe":"single_train_fitted_shrinkage",**scores(pred,p['y'],p['persistence'])})
            base_rmse=scores(prediction['val'],val['y'],val['persistence'])['RMSE']
            probes.append({"condition":condition,"seed":seed,"probe":"bias_only","features":0,
                           **scores(prediction['val']+intercept,val['y'],val['persistence']),"baseline_val_RMSE":base_rmse})
            for feature,(z,inverse,mu,sd) in matrices.items():
                coefficients=inverse@(residual-intercept)
                corrected=prediction['val']+intercept+((features['val'][feature]-mu)/sd)@coefficients
                probes.append({"condition":condition,"seed":seed,"probe":feature,"features":len(mu),
                               **scores(corrected,val['y'],val['persistence']),"baseline_val_RMSE":base_rmse})
    save('residual_decomposition.csv',rows);save('lag_diagnosis.csv',lags)
    save('residual_information_probes.csv',probes);save('train_fitted_shrinkage.csv',shrink)
    summaries=pd.DataFrame(rows).groupby(['condition','split','group']).mean(numeric_only=True)
    summaries.to_csv(OUT/'residual_summary.csv',encoding='utf-8-sig')
    pd.DataFrame(probes).groupby(['condition','probe'])[['RMSE','MAE','baseline_val_RMSE']].mean().to_csv(OUT/'probe_summary.csv',encoding='utf-8-sig')
    dump(OUT/'frozen_diagnosis.json',{'status':'completed','thresholds_m':thresholds.tolist(),'test_evaluated':False,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'probe_ridge_penalty':.001,'probe_fit':'train residuals only; already-trained backbone, not out-of-fold predictions',
        'interpretation':'exploratory residual-information probes; not fair benchmark competitors or proof of causality/novelty',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(summaries.loc[('LastAnchor_wd0','val')][['bias','RMSE','predicted_change_RMS','true_change_RMS','excess_MSE_over_persistence']].to_string())
    print(pd.DataFrame(probes).groupby(['condition','probe'])[['RMSE','baseline_val_RMSE']].mean().to_string())


if __name__=='__main__':
    main()
