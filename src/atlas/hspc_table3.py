"""Table-3 HSPC three-modality (Mc/Mu/Ms) T3/T4 experimental runner.

Adds an INDEPENDENT workflow; does not mutate Table-2 trainer or the E1-E4 jobs.
T3: learned TimeHead distilled from label-free graph/diff/attribution teacher.
T4: learned latent velocity + forward-time kNN transition + train-only terminal absorption.

CRITICAL: T4 y_true is a cell-type-derived fate PROXY, not measured future fate.
Training-stage annotations choose only a time sign; test labels never select hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import random
import hashlib
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ATLAS
from .lag import build_lag_banks
from .temporal import build_temporal_teacher
from .losses import (
    reconstruction_and_kl, multilevel_alignment_loss, biological_structure_loss,
    lag_objective, time_distillation_loss, paired_contrastive_loss,
)

DEFAULT_STAGE_ORDER = {
    'HSC': 0, 'LMPP': 1, 'CMP': 1, 'GMP': 1, 'MEP': 1,
    'Prog B': 1, 'Prog DC': 1, 'Prog MK': 1,
    'DC': 2, 'Granulocyte': 2, 'Erythrocyte': 2,
    'Megakaryocyte': 2, 'Platelet': 2, 'Mast Cells': 2,
}
DEFAULT_FUTURE_MAP = {
    'LMPP': 'lymphoid_dc', 'Prog B': 'lymphoid_dc', 'Prog DC': 'lymphoid_dc', 'DC': 'lymphoid_dc',
    'CMP': 'myeloid', 'GMP': 'myeloid', 'Granulocyte': 'myeloid', 'Mast Cells': 'myeloid',
    'MEP': 'ery_platelet', 'Prog MK': 'ery_platelet', 'Megakaryocyte': 'ery_platelet',
    'Platelet': 'ery_platelet', 'Erythrocyte': 'ery_platelet',
}
DEFAULT_TERMINAL_TYPES = ['DC', 'Granulocyte', 'Erythrocyte', 'Megakaryocyte', 'Platelet', 'Mast Cells']
DEFAULT_EXCLUDE_TYPES = ['HSC']
DATASET_FILES = {
    'HSPC_2M': ('3423-MV-2-8489-MV-1_adata_postpro_concat.h5ad',
                '3423-MV-2-8489-MV-1_adata_atac_postpro_concat.h5ad'),
    'HSPC_RNA': ('3423-MV-2-8489-MV-1-GSM5460410_adata_postpro_concat.h5ad',
                 '3423-MV-2-8489-MV-1-GSM5460410_adata_atac_postpro_concat.h5ad'),
    'HSPC_MAC': ('8489-MV-1-9060-MV-3_adata_postpro_concat.h5ad',
                 '8489-MV-1-9060-MV-3_adata_atac_postpro_concat.h5ad'),
}
EDGES = [('atac', 'unspliced'), ('unspliced', 'spliced'), ('atac', 'spliced')]


@dataclass
class HSPCConfig:
    dataset: str = 'HSPC_2M'
    data_root: str = '/data5/zhangye/VelocityBench/data'
    output_root: str = '/data5/zhangye/ATLAS/results/table3_hspc'
    label_key: str = 'celltype'
    seed: int = 1234
    device: str = 'cuda:0'
    top_n_genes: int = 2000
    test_fraction: float = 0.20
    batch_size: int = 256
    epochs_phase1: int = 80
    epochs_phase2: int = 40
    epochs_phase3: int = 60
    learning_rate: float = 1e-3
    shared_dim: int = 32
    private_dim: int = 16
    refresh_teacher_every: int = 10
    teacher_landmark_cells: int = 4000
    k_neighbors: int = 15
    lag_bins: int = 4
    lambda_mmd: float = 0.05
    lambda_ot: float = 0.20
    lambda_bio: float = 0.50
    lambda_time: float = 1.0
    lambda_lag: float = 0.30
    lambda_velocity: float = 0.20
    lambda_cross_pred: float = 0.5
    max_ot_cells: int = 128
    max_transition_neighbors: int = 64
    transition_steps: int = 60
    transition_beta_velocity: float = 0.7
    transition_beta_time: float = 2.0
    quick: bool = False


@dataclass
class HSPCData:
    x: Dict[str, np.ndarray]
    cell_ids: np.ndarray
    labels: np.ndarray
    stages: np.ndarray
    fate_labels: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    terminal: np.ndarray
    eval_mask: np.ndarray
    gene_names: np.ndarray
    metadata: dict


def json_dump(d, p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    def clean(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        if isinstance(o, float) and not np.isfinite(o): return None
        raise TypeError(type(o).__name__)
    p.write_text(json.dumps(d, default=clean, indent=2, ensure_ascii=False))


def _variance(x):
    if sparse.issparse(x):
        m = np.asarray(x.mean(axis=0)).reshape(-1)
        second = np.asarray(x.multiply(x).mean(axis=0)).reshape(-1)
        return np.maximum(second - m ** 2, 0)
    return np.var(np.asarray(x), axis=0)


def _dense(x):
    return (x.toarray() if sparse.issparse(x) else np.asarray(x)).astype(np.float32, copy=False)


def label_holdout_split(labels, test_fraction=.2, seed=1234):
    """Celltype-stratified split; labels here used only to allocate split, not in model losses."""
    rng = np.random.default_rng(seed)
    ids_train, ids_test = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_test = max(1, min(len(idx) - 2, int(round(len(idx) * test_fraction)))) if len(idx) >= 4 else 0
        ids_test.extend(idx[:n_test].tolist()); ids_train.extend(idx[n_test:].tolist())
    train = np.sort(np.asarray(ids_train, dtype=int))
    test = np.sort(np.asarray(ids_test, dtype=int))
    if len(test) < 10:
        raise ValueError('Too few held-out cells after stratified split')
    if np.intersect1d(train, test).size: raise AssertionError('Split leakage')
    return train, test


def load_hspc_data(cfg: HSPCConfig, annotation_config: dict | None = None) -> HSPCData:
    """Read old V4 HSPC paths with exact barcode/gene alignment and train-only scaling."""
    try:
        import anndata as ad
    except ImportError as e:
        raise ImportError('Install anndata: pip install anndata') from e
    if cfg.dataset not in DATASET_FILES: raise KeyError(cfg.dataset)
    filenames = DATASET_FILES[cfg.dataset]
    rna_path, atac_path = (Path(cfg.data_root) / f for f in filenames)
    for p in (rna_path, atac_path):
        if not p.exists(): raise FileNotFoundError(f'Missing HSPC source: {p}')
    rna, atac = ad.read_h5ad(rna_path), ad.read_h5ad(atac_path)
    for mat, layers in ((rna, ('Mu', 'Ms')), (atac, ('Mc',))):
        for lay in layers:
            if lay not in mat.layers: raise KeyError(f'Missing {lay} in {mat}; present: {list(mat.layers)}')
    if not rna.obs_names.is_unique or not atac.obs_names.is_unique:
        raise ValueError('Duplicate cell IDs in source files')
    cells = rna.obs_names.intersection(atac.obs_names, sort=False)
    genes = rna.var_names.intersection(atac.var_names, sort=False)
    if len(cells) < 30 or len(genes) < 10:
        raise ValueError(f'Too little overlap: cells={len(cells)}, genes={len(genes)}')
    if cfg.label_key not in rna.obs:
        raise KeyError(f'Expected label_key={cfg.label_key}; options={list(rna.obs.columns)}')
    # Limit loaded matrices by selecting barcode and gene intersection BEFORE dense conversion.
    rna = rna[cells, genes].copy()
    atac = atac[cells, genes].copy()
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise AssertionError('RNA/ATAC cell order mismatch')
    labels = rna.obs[cfg.label_key].astype(str).to_numpy()
    train_idx, test_idx = label_holdout_split(labels, cfg.test_fraction, cfg.seed)
    # Gene selection uses train portion ONLY; no test-expression leakage.
    vari = sum((_variance(mat[train_idx]) for mat in
                (rna.layers['Mu'], rna.layers['Ms'], atac.layers['Mc'])))
    if cfg.top_n_genes and cfg.top_n_genes < len(genes):
        feature_ids = np.sort(np.argsort(vari)[-cfg.top_n_genes:])
    else:
        feature_ids = np.arange(len(genes))
    sel_genes = np.asarray(genes)[feature_ids].astype(str)
    xs_raw = {
        'unspliced': _dense(rna.layers['Mu'][:, feature_ids]),
        'spliced': _dense(rna.layers['Ms'][:, feature_ids]),
        'atac': _dense(atac.layers['Mc'][:, feature_ids]),
    }
    x = {}
    for m, v in xs_raw.items():
        mu = v[train_idx].mean(0)
        sd = v[train_idx].std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        x[m] = np.nan_to_num((v - mu[None, :]) / sd[None, :], nan=0, posinf=0, neginf=0).astype(np.float32)
        if not np.isfinite(x[m]).all():raise ValueError(f'Non-finite {m}')
    annotation_config = annotation_config or {}
    stage_map = annotation_config.get('stage_order_map', DEFAULT_STAGE_ORDER)
    future_map = annotation_config.get('future_map', DEFAULT_FUTURE_MAP)
    terminal_types = set(annotation_config.get('terminal_types', DEFAULT_TERMINAL_TYPES))
    exclude_types = set(annotation_config.get('exclude_eval_types', DEFAULT_EXCLUDE_TYPES))
    stages = np.asarray([stage_map.get(k, np.nan) for k in labels], dtype=float)
    fates = np.asarray([str(future_map.get(k, '')) for k in labels], dtype=object)
    terminal = np.isin(labels, list(terminal_types))
    eligible = (~terminal) & (~np.isin(labels, list(exclude_types))) & (fates != '')
    unmapped = sorted(set(labels) - set(stage_map))
    unfated = sorted(set(labels) - set(future_map))
    metadata = {
        'input_rna': str(rna_path), 'input_atac': str(atac_path),
        'n_cells': len(cells), 'n_genes': len(sel_genes),
        'n_train': len(train_idx), 'n_test': len(test_idx),
        'label_distribution': pd.Series(labels).value_counts().to_dict(),
        'unmapped_stage_labels': unmapped, 'unmapped_fate_labels': unfated,
        'stage_order_map': stage_map, 'future_map': future_map,
        'terminal_types': sorted(terminal_types), 'exclude_eval_types': sorted(exclude_types),
        'annotation_provenance': 'V2/V4 legacy notebook defaults or user-provided file; review per cohort',
        't4_target_provenance': 'CELLTYPE_DERIVED_FATE_PROXY_NOT_EXPERIMENTAL_FUTURE',
    }
    print(f"[{cfg.dataset}] cells={len(cells)} genes={len(sel_genes)} train={len(train_idx)} test={len(test_idx)}")
    print('Unmapped stage labels:', unmapped)
    print('Unmapped fate labels:', unfated)
    return HSPCData(x, np.asarray(cells).astype(str), labels, stages, fates, train_idx,
                    test_idx, terminal, eligible, sel_genes, metadata)


def _latent_numpy(model, data, ids, device, chunk=512):
    model.eval()
    hs, ts, zs = [], [], {m: [] for m in sorted(data.x)}
    with torch.no_grad():
        for sub in np.array_split(ids, max(1, int(np.ceil(len(ids)/chunk)))):
            v = {m: torch.from_numpy(x[sub]).to(device) for m,x in data.x.items()}
            masks = {m: torch.ones(len(sub), device=device) for m in v}
            out = model(v, masks, sample=False)
            hs.append(out['h'].cpu().numpy()); ts.append(out['time'].view(-1).cpu().numpy())
            for m in zs: zs[m].append(out['enc'][m]['mu_shared'].cpu().numpy())
    return np.concatenate(hs), np.concatenate(ts), {m:np.concatenate(v) for m,v in zs.items()}


def _teacher(model, data, cfg, device):
    # Landmark teacher avoids running spectral embedding + shortest path on entire cohort.
    hs, _, zs = _latent_numpy(model, data, data.train_idx, device)
    n = len(hs)
    rng = np.random.default_rng(cfg.seed + 17)
    landmark = np.sort(rng.choice(n, size=min(n, cfg.teacher_landmark_cells), replace=False))
    h_sub = hs[landmark]
    masks = {m: np.ones(len(landmark),dtype=float) for m in zs}
    t = build_temporal_teacher(
        h=h_sub, shared={m: z[landmark] for m,z in zs.items()}, masks=masks,
        k_neighbors=cfg.k_neighbors, tau=.85, lambda_sm=1., lambda_mv=1.,lambda_sp=1.,
        target_std=.2,method='local_consensus',consensus_tau=.15,
        lambda_agreement=3.,lambda_local_smooth=1.,
    )
    if len(landmark)==n:
        teacher=np.asarray(t.teacher,dtype=np.float32)
    else:
        nn_search=NearestNeighbors(n_neighbors=min(5,len(landmark))).fit(h_sub)
        dist, idx=nn_search.kneighbors(hs)
        w=1/(dist+1e-5);w/=w.sum(1,keepdims=True)
        teacher=np.sum(w*t.teacher[idx],axis=1).astype(np.float32)
    lag_banks=build_lag_banks(zs, {m: np.ones(n,bool) for m in zs},teacher,
                              edges=EDGES,n_bins=cfg.lag_bins)
    # velocity pseudo-target in shared latent, using later time-compatible close neighbors.
    nn_search=NearestNeighbors(n_neighbors=min(max(25,cfg.k_neighbors*4),n)).fit(hs)
    dist,idx=nn_search.kneighbors(hs)
    time_deltas=teacher[idx]-teacher[:,None]
    valid=time_deltas>0.005
    cost=dist + 0.30 * np.maximum(time_deltas-0.12,0)
    cost[~valid]=np.inf
    nxt=idx[np.arange(n),np.argmin(cost,axis=1)]
    has_next=valid.any(1)
    velocity_target=(hs[nxt]-hs).astype(np.float32)
    velocity_target[~has_next]=0
    return teacher,lag_banks,velocity_target,has_next, t


def train_hspc(cfg: HSPCConfig, data: HSPCData, output_dir: Path):
    """Three-phase ATLAS training without stage/fate supervision in gradients."""
    from .utils import seed_everything
    seed_everything(cfg.seed)
    device=torch.device(cfg.device if not cfg.device.startswith('cuda') or torch.cuda.is_available() else 'cpu')
    if cfg.device.startswith('cuda') and device.type=='cpu':
        warnings.warn('CUDA unavailable, falling back to CPU')
    n_train=len(data.train_idx)
    train_tensor={m:torch.as_tensor(x[data.train_idx],dtype=torch.float32) for m,x in data.x.items()}
    dl=DataLoader(TensorDataset(torch.arange(n_train)),batch_size=cfg.batch_size,
                  shuffle=True,num_workers=0)
    input_dims={m: x.shape[1] for m,x in data.x.items()}
    model=ATLAS(input_dims,shared_dim=cfg.shared_dim,private_dim=cfg.private_dim,
                lag_edges=EDGES).to(device)
    velocity_head=nn.Sequential(nn.Linear(cfg.shared_dim,64),nn.GELU(),
                                nn.Linear(64,cfg.shared_dim)).to(device)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(velocity_head.parameters()),
                                lr=cfg.learning_rate,weight_decay=1e-5)
    history=[];cache=None
    phase_list=[('phase1',cfg.epochs_phase1),('phase2',cfg.epochs_phase2),('phase3',cfg.epochs_phase3)]
    if cfg.quick:phase_list=[('phase1',2),('phase2',1),('phase3',2)]
    for phase, epochs in phase_list:
        for epoch in range(epochs):
            if phase=='phase3' and (cache is None or epoch%cfg.refresh_teacher_every==0):
                model.eval()
                cache=_teacher(model,data,cfg,device)
                print(f'[teacher] phase3 epoch={epoch+1}, anchors={min(n_train,cfg.teacher_landmark_cells)}')
            model.train();velocity_head.train()
            means={};nb=0
            for (j,) in dl:
                idx=j.numpy();n=len(idx)
                x={m: v[j].to(device) for m,v in train_tensor.items()}
                masks={m:torch.ones(n,dtype=torch.float32,device=device) for m in x}
                lag_bank, lag_valid=None,None
                if phase=='phase3':
                    lag_bank={edge:torch.from_numpy(z.bank[idx]).to(device) for edge,z in cache[1].items()}
                    lag_valid={edge:torch.from_numpy(z.valid[idx]).to(device) for edge,z in cache[1].items()}
                out=model(x,masks,lag_banks=lag_bank,lag_valid=lag_valid)
                rec,kl=reconstruction_and_kl(out,x,masks)
                # Match the current Table-2 Phase-I objective: reconstruction, KL, paired contrastive.
                modality_names=sorted(out['shared'])
                contrasts=[]
                for ai,am in enumerate(modality_names):
                    for bm in modality_names[ai+1:]:
                        contrasts.append(paired_contrastive_loss(
                            out['shared'][am],out['shared'][bm],masks[am].bool() & masks[bm].bool()))
                contrast=sum(contrasts)/max(1,len(contrasts))
                loss=rec+0.001*kl+contrast
                if phase != 'phase1':
                    mmd,ot,con=multilevel_alignment_loss(out,masks,epsilon=.05,n_iter=20,
                                                         max_cells=cfg.max_ot_cells)
                    bio=biological_structure_loss(out,x,masks,max_cells=cfg.max_ot_cells)
                    # Contrastive is already accounted for in every phase above.
                    loss=loss+cfg.lambda_mmd*mmd+cfg.lambda_ot*ot+cfg.lambda_bio*bio
                    pred_loss=out['h'].new_zeros(())
                    for edge in EDGES:
                        k=f'{edge[0]}__to__{edge[1]}'
                        pred_loss=pred_loss+torch.nn.functional.mse_loss(out['cross'][k],x[edge[1]])
                    loss=loss+cfg.lambda_cross_pred*(pred_loss/len(EDGES))
                if phase=='phase3':
                    tt=torch.as_tensor(cache[0][idx],dtype=out['time'].dtype,device=device)
                    t_loss,t_reg=time_distillation_loss(out['time'].view(-1),tt,out['h'],target_std=.20)
                    lag_loss=out['h'].new_zeros(())
                    for edge in EDGES:
                        lr,_=lag_objective(out,x,masks,tt,source=edge[0],target=edge[1],
                                          lambda_unpaired=1.,temporal_window=.35,local_k=24)
                        lag_loss=lag_loss+lr
                    vel_target=torch.as_tensor(cache[2][idx],device=device)
                    valid_vel=torch.as_tensor(cache[3][idx],device=device)
                    v_pred=velocity_head(out['h'])
                    vel_loss=torch.nn.functional.mse_loss(v_pred[valid_vel],vel_target[valid_vel]) if bool(valid_vel.any()) else v_pred.sum()*0
                    lag_scale=min(1.,(epoch+1)/5.)
                    loss=loss+cfg.lambda_time*(t_loss+.1*t_reg)+lag_scale*cfg.lambda_lag*(lag_loss/len(EDGES)) +cfg.lambda_velocity*vel_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(velocity_head.parameters()),5.0)
                optimizer.step()
                means['loss']=means.get('loss',0.)+float(loss.item());nb+=1
            row={'phase':phase,'epoch':epoch+1,'loss':means['loss']/max(1,nb)}
            history.append(row)
            print(f"[{cfg.dataset}] {phase} {epoch+1}/{epochs} loss={row['loss']:.4f}")
        # Save completed phases (not a label-selected checkpoint).
        output_dir.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'velocity_head':velocity_head.state_dict(),
                    'config':asdict(cfg),'phase':phase},output_dir/f'checkpoint_{phase}_last.pt')
        pd.DataFrame(history).to_csv(output_dir/'training_history.csv',index=False)
    return model,velocity_head,device


def _ordering_metrics(t, stages):
    """O(K^2*NlogN) equivalent to per-stage AUROC; weighted pairwise POA differs
    from legacy only for exact time ties (legacy ties count as 0 vs ROC 0.5)."""
    keep=np.isfinite(t)&np.isfinite(stages)
    t=np.asarray(t)[keep]; s=np.asarray(stages)[keep]
    uniq=np.sort(np.unique(s))
    num_poa=den_poa=num_auc=den_auc=0.
    for i,a in enumerate(uniq):
        for b in uniq[i+1:]:
            v=(s==a)|(s==b)
            if min(np.sum(s[v]==a), np.sum(s[v]==b))<1: continue
            auc=roc_auc_score((s[v]==b).astype(int),t[v]);gap=abs(b-a)
            num_auc+=gap*auc;den_auc+=gap
            pairs=int(np.sum(s[v]==a))*int(np.sum(s[v]==b))
            num_poa+=gap*pairs*auc;den_poa+=gap*pairs
    return {'POA':num_poa/den_poa if den_poa else np.nan,
            'POAUC':num_auc/den_auc if den_auc else np.nan,
            'n_eval':int(len(t)), 'stages':int(len(uniq))}


def _row_softmax(x):
    x=np.asarray(x,dtype=np.float64)
    x=x-np.max(x)
    z=np.exp(x)
    return z/max(z.sum(),1e-20)


def transition_fate_scores(reference_z,reference_time,reference_velocity,
                           reference_terminal,reference_fate,query_z,query_time,query_velocity,
                           classes,neighbors=64,steps=60,beta_vel=.7,beta_time=2.):
    """Train-reference-only Markov chain with absorbing labeled TRAIN terminal nodes.
    No test cell appears in the transition reference, prototypes or label calibration."""
    n=len(reference_z);c=len(classes)
    if n<4 or c<2: raise ValueError('Need >=4 training references and >=2 fates')
    class_map={cl:i for i,cl in enumerate(classes)}
    terminal_code=np.array([class_map.get(str(x),-1) for x in reference_fate])
    terminal=reference_terminal & (terminal_code>=0)
    for k in range(c):
        if not np.any(terminal & (terminal_code==k)):
            raise ValueError(f'Missing training terminal reference for class {classes[k]}')
    k=min(n,neighbors+1)
    nn_ref=NearestNeighbors(n_neighbors=k).fit(reference_z)
    d,idx=nn_ref.kneighbors(reference_z)
    rows=[];cols=[];weights=[]
    def append_neighbors(node, candidates, distances,z,t,velocity,allow_self=False):
        if allow_self and reference_terminal[node]:
            rows.append(node);cols.append(node);weights.append(1.);return
        valid=(t[candidates]>t[node]+1e-4)
        valid &= candidates!=node
        if not np.any(valid):
            rows.append(node);cols.append(node);weights.append(1.);return
        cand=candidates[valid]; dd=distances[valid]
        disp=z[cand]-z[node]
        cn=(disp*velocity[node]).sum(-1)/(np.linalg.norm(disp,axis=1)*np.linalg.norm(velocity[node])+1e-8)
        score=-dd/(np.median(dd)+1e-4) + beta_vel*cn + beta_time*(t[cand]-t[node])
        w=_row_softmax(score)
        rows.extend([node]*len(cand));cols.extend(cand.tolist());weights.extend(w.tolist())
    for i in range(n):
        if terminal[i]:
            rows.append(i);cols.append(i);weights.append(1.)
        else:
            append_neighbors(i,idx[i],d[i],reference_z,reference_time,reference_velocity)
    P=sparse.csr_matrix((weights,(rows,cols)),shape=(n,n),dtype=np.float64)
    F=np.zeros((n,c),dtype=np.float64)
    F[terminal,terminal_code[terminal]]=1.
    fixed=F.copy()
    for _ in range(steps):
        F=P@F
        F[terminal]=fixed[terminal]
    # Query -> train reference, no test identities/labels appear in reference.
    nq=len(query_z)
    qd, qi=nn_ref.kneighbors(query_z,n_neighbors=min(n,neighbors))
    scores=np.zeros((nq,c),dtype=np.float64)
    for i in range(nq):
        valid=reference_time[qi[i]] > query_time[i]+1e-4
        # fallback: nearest references when time-neighbor graph is disconnected
        candidates=qi[i][valid] if valid.any() else qi[i]
        dist=qd[i][valid] if valid.any() else qd[i]
        disp=reference_z[candidates]-query_z[i]
        cos=(disp*query_velocity[i]).sum(-1)/(np.linalg.norm(disp,axis=1)*np.linalg.norm(query_velocity[i])+1e-8)
        dt=reference_time[candidates]-query_time[i]
        logits=-dist/(np.median(dist)+1e-4)+beta_vel*cos+beta_time*np.maximum(dt,0)
        scores[i]=_row_softmax(logits)@F[candidates]
    reach=scores.sum(1)
    unresolved=np.sum(reach<1e-5)
    counts=np.bincount(terminal_code[terminal],minlength=c).astype(float)
    prior=counts/counts.sum()
    scores[reach<1e-5]=prior
    scores=np.clip(scores,1e-10,None)
    scores/=scores.sum(1,keepdims=True)
    return scores,{'n_reference':int(n),'n_terminal':int(terminal.sum()),
                   'n_queries':int(nq),'n_unresolved_prior_fallback':int(unresolved),
                   'transition_rows':int(P.shape[0]),'transition_nnz':int(P.nnz),
                   'terminal_reference_classes':{cl:int(np.sum(terminal & (terminal_code==k))) for k,cl in enumerate(classes)}}


def evaluate_hspc(cfg: HSPCConfig,data: HSPCData,model,velocity_head,device,output_dir:Path):
    # NO test stage/fate is read until model, velocity, transition and time direction are fixed.
    full_idx=np.arange(len(data.cell_ids))
    z,t,_=_latent_numpy(model,data,full_idx,device)
    velocity_head.eval()
    velocities=[]
    with torch.no_grad():
        for ids in np.array_split(full_idx,max(1,int(np.ceil(len(full_idx)/512)))):
            velocities.append(velocity_head(torch.tensor(z[ids],device=device,dtype=torch.float32)).cpu().numpy())
    velocities=np.concatenate(velocities)
    train_stage=data.stages[data.train_idx]
    ok=np.isfinite(train_stage)
    corr=spearmanr(t[data.train_idx][ok],train_stage[ok]).statistic if ok.sum()>5 else np.nan
    orientation=-1 if np.isfinite(corr) and corr<0 else 1
    orient_t=1-t if orientation==-1 else t.copy()
    orient_velocity=velocities *orientation
    test_ids=data.test_idx
    t3=_ordering_metrics(orient_t[test_ids],data.stages[test_ids])
    t3.update({'orientation_from_train_stage_only':int(orientation),
               'train_stage_spearman_for_orientation':float(corr) if np.isfinite(corr) else None,
               'time_source':'trained_time_head',
               'reference_labels_for_training_loss':False,'test_labels_for_orientation':False})
    ref_ids=data.train_idx
    eval_ids=test_ids[data.eval_mask[test_ids]]
    train_terminal_fates=np.unique(data.fate_labels[ref_ids][data.terminal[ref_ids]])
    classes=sorted(set(train_terminal_fates))  # Training references ONLY; never derive candidates from test labels
    if len(classes)<2:raise ValueError(f'T4 requires >=2 fates represented by train terminals and eval queries; got {classes}')
    query=eval_ids[np.isin(data.fate_labels[eval_ids],classes)]
    scores, transition_info=transition_fate_scores(
        z[ref_ids],orient_t[ref_ids],orient_velocity[ref_ids],
        data.terminal[ref_ids],data.fate_labels[ref_ids],
        z[query],orient_t[query],orient_velocity[query],classes,
        neighbors=cfg.max_transition_neighbors,steps=cfg.transition_steps,
        beta_vel=cfg.transition_beta_velocity,beta_time=cfg.transition_beta_time)
    y=np.array([classes.index(x) for x in data.fate_labels[query]])
    if len(query)==0: raise ValueError('No held-out eligible T4 queries with training-reference fate classes')
    t4={'F1':float(f1_score(y,np.argmax(scores,axis=1),average='macro',labels=np.arange(len(classes)))),
        'AUROC':None,'n_eval':int(len(query)),'classes':classes,
        'target':'celltype_derived_fate_proxy', 'fate_labels_used_to_train_velocity':False,
        'heldout_eval_cells_overlap_terminal_reference':False}
    try:
        if len(set(y))==len(classes):
            t4['AUROC']=float(roc_auc_score(y,scores[:,1]) if len(classes)==2 else roc_auc_score(y,scores,multi_class='ovr',average='macro',labels=np.arange(len(classes))))
    except ValueError as exc:
        t4['auc_warning']=str(exc)
    # Baseline: same trained model, replace directional velocity bias with zero.
    zero=np.zeros_like(orient_velocity)
    baseline,_=transition_fate_scores(
        z[ref_ids],orient_t[ref_ids],zero[ref_ids],data.terminal[ref_ids],data.fate_labels[ref_ids],
        z[query],orient_t[query],zero[query],classes,neighbors=cfg.max_transition_neighbors,
        steps=cfg.transition_steps,beta_vel=0.,beta_time=cfg.transition_beta_time)
    t4['time_only_F1']=float(f1_score(y,np.argmax(baseline,axis=1),average='macro',labels=np.arange(len(classes))))
    if len(set(y))==len(classes):
        t4['time_only_AUROC']=float(roc_auc_score(y,baseline[:,1]) if len(classes)==2 else roc_auc_score(y,baseline,multi_class='ovr',average='macro',labels=np.arange(len(classes))))
    output_dir.mkdir(parents=True,exist_ok=True)
    json_dump(t3,output_dir/'T3_metrics.json')
    json_dump(t4,output_dir/'T4_metrics.json')
    json_dump(transition_info,output_dir/'T4_transition_diagnostics.json')
    pd.DataFrame({'cell_id':data.cell_ids[query], 'true_fate_proxy':data.fate_labels[query],
                  **{f'p_{name}':scores[:,i] for i,name in enumerate(classes)}}).to_csv(output_dir/'T4_heldout_predictions.csv',index=False)
    np.savez_compressed(output_dir/'T3_test_predictions.npz',cell_ids=data.cell_ids[test_ids],
                        time=orient_t[test_ids],stage=data.stages[test_ids],h=z[test_ids])
    plot_trajectory(z[test_ids],orient_t[test_ids],data.stages[test_ids], output_dir/'T3_trajectory.png')
    return t3,t4,transition_info


def plot_trajectory(z,t,stage,filepath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    try:
        from umap import UMAP
        xy=UMAP(n_neighbors=25,min_dist=.3,random_state=1234).fit_transform(z)
        xlabel='UMAP 1';ylabel='UMAP 2'
    except ImportError:
        from sklearn.decomposition import PCA
        xy=PCA(n_components=2,random_state=1234).fit_transform(z)
        xlabel='PCA 1';ylabel='PCA 2'
    cmap=LinearSegmentedColormap.from_list('bluepink',['#6FA8DC','#F08AC0'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    p=axes[0].scatter(xy[:,0],xy[:,1],c=t,s=3,cmap=cmap,vmin=0,vmax=1,linewidths=0)
    fig.colorbar(p,ax=axes[0],label='Predicted time')
    q=axes[1].scatter(xy[:,0],xy[:,1],c=stage,s=3,cmap='coolwarm',linewidths=0)
    fig.colorbar(q,ax=axes[1],label='Reference stage')
    for ax,title in zip(axes,['ATLAS predicted time (held-out)','Reference stages (held-out)']):
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
        ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(filepath,dpi=220);plt.close(fig)


def run_hspc(cfg: HSPCConfig, annotation_config:dict|None=None):
    if cfg.dataset not in DATASET_FILES:raise KeyError(cfg.dataset)
    out=Path(cfg.output_root)/cfg.dataset/('quick' if cfg.quick else 'full')/f'seed_{cfg.seed}'
    if (out/'T4_metrics.json').exists():
        raise FileExistsError(f'Finished output exists: {out}; choose a new seed or output_root')
    out.mkdir(parents=True,exist_ok=True)
    json_dump(asdict(cfg),out/'config.json')
    data=load_hspc_data(cfg,annotation_config)
    json_dump(data.metadata,out/'data_audit.json')
    # Entire split documented for reproducibility and leakage checks.
    pd.DataFrame({'cell_id':data.cell_ids,'partition':np.where(np.isin(np.arange(len(data.labels)),data.train_idx),'train','test'),
                  'cell_type':data.labels, 'stage':data.stages,'fate_proxy':data.fate_labels,
                  'terminal':data.terminal,'t4_eligible':data.eval_mask}).to_csv(out/'split_manifest.csv',index=False)
    model,velocity,device=train_hspc(cfg,data,out)
    t3,t4,transition_info=evaluate_hspc(cfg,data,model,velocity,device,out)
    final={'dataset':cfg.dataset,'seed':cfg.seed,'T3':t3,'T4':t4,
           'annotation_protocol':'train-only stage orientation; train-only terminal prototypes; heldout test',
           'warning':'Fate proxy from celltype, NOT real future longitudinal measurement; not directly comparable to paper Table 3'}
    json_dump(final,out/'summary.json')
    print(json.dumps(final,indent=2,ensure_ascii=False))
    return final,out
