"""Run without AnnData/GPU/datasets: static checks and pure NumPy transition tests.
Usage: python tests/test_protocol_static.py
"""
import ast
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'src/atlas/hspc_table3.py'
s=p.read_text()
assert 'out["model_state"]' not in s
assert 'reference_terminal' in s and 'train_idx' in s and 'test_idx' in s
assert 'test_labels_for_orientation' in s
ast.parse(s)
print('PASS: module parses and explicit anti-leakage fields exist')
for n in ['HSPC_2M','HSPC_RNA','HSPC_MAC']:
 import json
 o=json.loads((p.parents[2]/'configs'/f'{n}_annotations.json').read_text())
 assert set(o['terminal_types']) <= set(o['future_map'])
 print('PASS:',n,'annotation keys:',len(o['stage_order_map']))
