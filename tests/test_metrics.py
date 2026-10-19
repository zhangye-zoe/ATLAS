import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas.metrics import evaluate_t1_coral_style, pairwise_ordering_accuracy


def _reference_coral_t1(pred, true, seed=1234):
    mix=np.vstack([np.asarray(true,dtype=float),np.asarray(pred,dtype=float)])
    nc=min(30,mix.shape[0]-1,mix.shape[1])
    emb=PCA(n_components=nc, random_state=seed).fit_transform(mix)
    n=len(true); a=emb[:n]; b=emb[n:]
    d=pairwise_distances(a,b)
    fos=[]; ped=[]
    for i in range(n):
        td=d[i,i]
        fos.append(np.mean(d[i] < td))
        ped.append(td)
    return float(np.mean(fos)), float(np.mean(ped))


def test_t1_matches_original_coral_protocol_exactly():
    rng=np.random.default_rng(7)
    true=rng.normal(size=(12,8))
    pred=true + rng.normal(scale=.25,size=true.shape)
    ref_fos,ref_ped=_reference_coral_t1(pred,true)
    out,_,_=evaluate_t1_coral_style(pred,true,seed=1234)
    assert np.isclose(out['FOSCTTM'],ref_fos)
    assert np.isclose(out['PED'],ref_ped)


def test_poa_ties_are_incorrect_under_strict_indicator():
    time=np.asarray([0.5,0.5]); stages=np.asarray([0.0,1.0])
    assert np.isclose(pairwise_ordering_accuracy(time,stages),0.0)
