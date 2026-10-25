"""Synthetic no-training test: import module with external ATLAS dependencies stubbed."""
from pathlib import Path
import importlib.util
import sys
import types
import numpy as np

this=Path(__file__).resolve().parents[1]/'src'/'atlas'/'hspc_table3.py'
base=types.ModuleType('atlas');base.__path__=[];sys.modules['atlas']=base
for name,attrs in {
    'model':dict(ATLAS=object),'lag':dict(build_lag_banks=lambda *a,**k:None),
    'temporal':dict(build_temporal_teacher=lambda *a,**k:None),
    'losses':dict(reconstruction_and_kl=None,multilevel_alignment_loss=None,
                  biological_structure_loss=None,lag_objective=None,time_distillation_loss=None,paired_contrastive_loss=None),
}.items():
    x=types.ModuleType('atlas.'+name);x.__dict__.update(attrs);sys.modules[x.__name__]=x
spec=importlib.util.spec_from_file_location('atlas.hspc_table3',this)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
assert mod.DATASET_FILES['HSPC_MAC'][0].startswith('8489-')
assert np.isclose(mod._ordering_metrics(np.array([0.,.1,.8,1.]),np.array([0,0,2,2]))['POAUC'],1)
train_z=np.array([[0,0],[1,0],[2,0],[4,0],[0,1],[1,1],[2,1],[4,1]],float)
t=np.array([0.05,.2,.4,.9,.05,.2,.4,.9])
v=np.array([[1,0]]*8,float)
term=np.array([0,0,0,1,0,0,0,1],bool)
fate=np.array(['A']*4+['B']*4)
query=np.array([[1.5,.1],[1.5,.9]])
qt=np.array([.3,.3]);qv=np.array([[1,0],[1,0]])
score,diag=mod.transition_fate_scores(train_z,t,v,term,fate,query,qt,qv,['A','B'],neighbors=5,steps=20)
assert score.shape==(2,2)
assert np.allclose(score.sum(1),1)
assert np.isfinite(score).all()
assert diag['n_terminal']==2
print('PASS synthetic T3 metric and T4 train-reference absorption:',score,diag)
