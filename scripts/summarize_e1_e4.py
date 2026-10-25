#!/usr/bin/env python3
"""Summarize ATLAS E1-E4 controlled ablations and paired differences from E1.

No test-set-based checkpoint selection or score optimization is performed here.
The script only reads saved metrics and verifies report-label partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    'FOSCTTM_pct': ('T1', 'FOSCTTM', 100.0),
    'PED': ('T1', 'PED', 1.0),
    'Pearson_pct': ('T2', 'Pearson', 100.0),
    'Pearson_gene_pct': ('T2', 'Pearson_gene', 100.0),
    'RMSE': ('T2', 'RMSE', 1.0),
    'POA_pct': ('T3', 'POA', 100.0),
    'POAUC_pct': ('T3', 'POAUC', 100.0),
}


def read_json(p):
    with p.open(encoding='utf-8') as f: return json.load(f)


def norm_float(x, factor=1):
    try: return float(x)*factor if x is not None else float('nan')
    except (TypeError,ValueError): return float('nan')


def label_partition_digest(path):
    if not path.is_file(): return None
    o=read_json(path)
    ids=o.get('report_cell_ids')
    if ids is None: return None
    return hashlib.sha256('\n'.join(sorted(map(str,ids))).encode()).hexdigest()[:16]


def gather(root):
    rows=[]
    for p in sorted(root.glob('E[1-4]/seed_*/[PB]*MC/pair_*/metrics.json')):
        # Glob [PB]*MC matches PBMC and BMMC, but filter defensively.
        ratio_dir=p.parent
        dataset=p.parent.parent.name.upper()
        if dataset not in {'PBMC','BMMC'}: continue
        arm=p.parents[3].name  # E1.../seed_1234/BMMC/pair_000/metrics.json: parents[3]=E1
        seed_dir=p.parents[2].name
        if arm not in {'E1','E2','E3','E4'} or not seed_dir.startswith('seed_'): continue
        m=read_json(p)
        es_file=ratio_dir/'early_stopping_summary.json'
        proto_file=ratio_dir/'e1_e4_experiment_protocol.json'
        es=read_json(es_file) if es_file.is_file() else {}
        pro=read_json(proto_file) if proto_file.is_file() else {}
        best=es.get('best') or {}
        selection=es.get('selection_policy') or pro.get('selection_policy')
        row={
            'experiment':arm, 'dataset':dataset,
            'seed':int(seed_dir.split('_')[-1]),
            'pairing_ratio':int(ratio_dir.name.rsplit('_',1)[-1]),
            'split_label':m.get('split_label'),
            'path':str(ratio_dir),
            'selection_policy':selection,
            'selected_phase':es.get('selected_phase',best.get('phase')),
            'selected_epoch':es.get('selected_epoch',best.get('phase_epoch')),
            'early_stopped':bool(es.get('early_stopped_phase3',False)),
            'phase3_frozen':bool(pro.get('freeze_representation_phase3',False)),
            'teacher_fixed':bool(pro.get('fixed_teacher_phase3',False)),
            't3_estimator':m.get('protocol',{}).get('T3_time_estimator'),
            'monitor_cells':m.get('protocol',{}).get('T3_monitor_cells'),
            'report_cells':m.get('protocol',{}).get('T3_report_cells'),
            'report_partition_hash':label_partition_digest(ratio_dir/'t3_validation_partition.json'),
            't3_n_eval':m.get('T3',{}).get('n_eval'),
        }
        for name,(task,field,factor) in METRICS.items():
            row[name]=norm_float(m.get(task,{}).get(field),factor)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No E1-E4 metrics.json files discovered under {root}')
    out=pd.DataFrame(rows)
    if out.duplicated(['dataset','seed','pairing_ratio','experiment']).any():
        raise ValueError('Duplicate result keys detected; check output paths')
    return out.sort_values(['dataset','pairing_ratio','seed','experiment']).reset_index(drop=True)


def validate_protocol(df):
    problems=[]
    for keys,group in df.groupby(['dataset','pairing_ratio','seed'],dropna=False):
        refs=group['report_partition_hash'].dropna().unique().tolist()
        if len(refs)>1: problems.append(f'{keys}: DIFFERENT held-out report labels {refs}')
        if len(refs)==0: problems.append(f'{keys}: missing held-out label manifest')
        estimators=group['t3_estimator'].dropna().unique().tolist()
        if len(estimators)>1: problems.append(f'{keys}: different T3 estimator {estimators}')
        evalcounts=group['t3_n_eval'].dropna().unique().tolist()
        if len(evalcounts)>1: problems.append(f'{keys}: different stage eval cohort size {evalcounts}')
    for _,r in df.iterrows():
        exp=r['experiment']; sel=r['selection_policy']; phase=r['selected_phase']
        if exp=='E1' and (sel!='final' or phase!='phase3'):
            problems.append(f'{exp} {r["dataset"]}/{r["pairing_ratio"]}: not final checkpoint')
        if exp!='E1' and (sel!='phase3' or phase!='phase3'):
            problems.append(f'{exp} {r["dataset"]}/{r["pairing_ratio"]}: NOT phase3-selected checkpoint')
        if exp=='E3' and not r['phase3_frozen']:
            problems.append(f'E3 {r["path"]}: no freeze')
        if exp=='E4' and not r['teacher_fixed']:
            problems.append(f'E4 {r["path"]}: no fixed teacher')
    if problems: raise ValueError('Invalid comparison protocol:\n - ' + '\n - '.join(problems))


def paired_vs_reference(df, reference="E1"):
    join_keys=['dataset','pairing_ratio','seed']
    base=df[df.experiment==reference].copy()
    if base.empty:
        return pd.DataFrame()
    measures=list(METRICS)
    use_cols=join_keys+measures+['report_partition_hash','t3_estimator']
    merged=df[df.experiment!=reference].merge(base[use_cols],on=join_keys,
                                            how='inner',suffixes=('',f'_{reference}'),validate='many_to_one')
    if merged.empty: return merged
    if not (merged["report_partition_hash"]==merged[f"report_partition_hash_{reference}"]).all():
        raise ValueError('E1 and E2-4 have different held-out label partitions')
    if not (merged["t3_estimator"]==merged[f"t3_estimator_{reference}"]).all():
        raise ValueError('E1 and E2-4 have different T3 estimators')
    # Positive = improvement for all these columns.
    for metric in measures:
        if metric in {'FOSCTTM_pct','PED','RMSE'}:
            merged[f'better_{metric}_vs_{reference}']=merged[f'{metric}_{reference}']-merged[metric]
        else:
            merged[f'better_{metric}_vs_{reference}']=merged[metric]-merged[f'{metric}_{reference}']
    return merged


def plot_summary(df, root):
    names=[('FOSCTTM_pct','T1 FOSCTTM (%) ↓'),('Pearson_pct','T2 Pearson (%) ↑'),
           ('RMSE','T2 RMSE ↓'),('POAUC_pct','T3 POAUC (%) ↑')]
    for dataset, sub in df.groupby('dataset'):
        fig, axs=plt.subplots(2,2,figsize=(11,7.0))
        for ax,(metric,label) in zip(axs.flatten(),names):
            for exp in ['E1','E2','E3','E4']:
                part=sub[sub.experiment==exp]
                if part.empty: continue
                agg=part.groupby('pairing_ratio')[metric].agg(['mean','std'])
                ax.plot(agg.index,agg['mean'],marker='o',label=exp)
                if agg['std'].notna().any():
                    std=agg['std'].fillna(0.0)
                    ax.fill_between(agg.index.to_numpy(),(agg['mean']-std).to_numpy(),
                                    (agg['mean']+std).to_numpy(),alpha=0.12)
            ax.set_xlabel('Paired cells (%)');ax.set_ylabel(label)
            ax.grid(alpha=.15)
        axs[0,0].legend(title='Experimental arm')
        fig.suptitle(f'ATLAS E1–E4 ablations on {dataset}',fontsize=15)
        fig.tight_layout()
        fig.savefig(root/f'{dataset}_e1_e4_metrics.png',dpi=200)
        fig.savefig(root/f'{dataset}_e1_e4_metrics.pdf')
        plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root',type=Path,required=True)
    args=ap.parse_args();root=args.results_root.resolve();root.mkdir(parents=True,exist_ok=True)
    df=gather(root)
    validate_protocol(df)
    df.to_csv(root/'all_e1_e4_runs.csv',index=False)
    fields=list(METRICS)
    summary=df.groupby(['dataset','pairing_ratio','experiment'],dropna=False)[fields].agg(['mean','std','count'])
    summary.columns=['_'.join(col) for col in summary.columns.to_flat_index()]
    summary.reset_index().to_csv(root/'summary_e1_e4.csv',index=False)
    paired=paired_vs_reference(df,'E1')
    if len(paired): paired.to_csv(root/'paired_vs_e1.csv',index=False)
    paired_e2=paired_vs_reference(df[df.experiment.isin(['E2','E3','E4'])],'E2')
    if len(paired_e2): paired_e2.to_csv(root/'paired_vs_e2.csv',index=False)
    plot_summary(df,root)
    print('Experiments:',df.experiment.value_counts().to_dict())
    print('Saved',root/'summary_e1_e4.csv')
    print('Saved',root/'all_e1_e4_runs.csv')
    if len(paired): print('Saved',root/'paired_vs_e1.csv')
    if len(paired_e2): print('Saved',root/'paired_vs_e2.csv')
    print('\nOne-seed results are exploratory; repeat with >=3 seeds before drawing conclusions.')

if __name__=='__main__': main()
