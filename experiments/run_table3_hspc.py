#!/usr/bin/env python3
"""Standalone command-line runner for the same module used by the 3 notebooks."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atlas.hspc_table3 import HSPCConfig, run_hspc, DATASET_FILES

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=DATASET_FILES, required=True)
p.add_argument('--data-root', default='/data5/zhangye/VelocityBench/data')
p.add_argument('--output-root', default='/data5/zhangye/ATLAS/results/table3_hspc')
p.add_argument('--annotation-json', default='')
p.add_argument('--seed', type=int, default=1234)
p.add_argument('--device', default='cuda:0')
p.add_argument('--top-n-genes', type=int, default=2000)
p.add_argument('--quick', action='store_true')
p.add_argument('--phase1', type=int, default=80)
p.add_argument('--phase2', type=int, default=40)
p.add_argument('--phase3', type=int, default=60)
args=p.parse_args()
config=HSPCConfig(dataset=args.dataset,data_root=args.data_root,output_root=args.output_root,
                  seed=args.seed,device=args.device,top_n_genes=args.top_n_genes,
                  epochs_phase1=args.phase1,epochs_phase2=args.phase2,epochs_phase3=args.phase3,quick=args.quick)
ann_path=Path(args.annotation_json) if args.annotation_json else ROOT/'configs'/f'{args.dataset}_annotations.json'
if not ann_path.is_absolute():ann_path=ROOT/ann_path
with ann_path.open() as f: ann=json.load(f)
_,out=run_hspc(config,ann)
print('Outputs:',out)
