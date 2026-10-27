"""Independent checks of saved outputs; skipped until run.py has been executed."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.core import arm_scores,policy_weights,policy_scores,summarize

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not all((ROOT/p).exists() for p in ['reports/results.json','data/processed/splits.csv','data/processed/holdout_predictions.csv']),reason="Run analysis first")

def test_splits_cover_every_row_exactly_once():
    s=pd.read_csv(ROOT/'data/processed/splits.csv')
    assert len(s)==64000
    assert s.source_row.is_unique
    assert set(s.source_row)==set(range(64000))
    assert s.split.value_counts().to_dict()=={'train':38400,'validation':12800,'test':12800}

def test_holdout_only_and_primary_contrast_recomputed():
    r=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    h=pd.read_csv(ROOT/'data/processed/holdout_predictions.csv')
    s=pd.read_csv(ROOT/'data/processed/splits.csv')
    assert set(h.source_row)==set(s.loc[s.split=='test','source_row'])
    assert len(h)==12800
    mu=h[['mu_0','mu_1','mu_2']].to_numpy()
    phi=arm_scores(h.conversion.to_numpy(),h.action.to_numpy(),mu)
    u=h[['multi_p_1','multi_p_2']].to_numpy()-h[['multi_p_0']].to_numpy()
    w=policy_weights(u.max(axis=1),.2,u.argmax(axis=1)+1)
    baseline=policy_weights(np.zeros(len(h)),.2,1,random=True)
    observed=summarize(policy_scores(w,phi)-policy_scores(baseline,phi))
    for key in ['estimate','se','low','high']:
        assert np.isclose(observed[key],r['primary'][key],atol=1e-12)

def test_equal_budgets_and_zero_contacts():
    p=pd.read_csv(ROOT/'reports/tables/test_policies.csv')
    assert np.allclose(p.coverage,p.actual_coverage)
    assert np.allclose(p.mens_share+p.womens_share,p.coverage)
    assert np.allclose(p.loc[p.coverage==0,['estimate','low','high']],0)
    for col in ['estimate','low','high','ipw_estimate']:
        assert p.loc[p.strategy=='uplift',col].iloc[-1]==pytest.approx(p.loc[p.strategy=='random_mens',col].iloc[-1])

