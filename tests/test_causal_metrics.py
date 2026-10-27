"""Проверки причинных оценок и бюджета на синтетических данных с известным эффектом."""
import numpy as np
import pytest
from src.core import arm_scores, policy_weights, policy_scores, summarize, holm, features

def test_zero_contact_has_exactly_zero_increment():
    rng=np.random.default_rng(11)
    y=rng.binomial(1,.2,1000)
    a=rng.integers(0,3,1000)
    scores=arm_scores(y,a,np.full((1000,3),.2))
    w=policy_weights(rng.normal(size=1000),0)
    assert np.array_equal(policy_scores(w,scores)-scores[:,0],np.zeros(1000))

def test_known_effect_recovered_and_paired_contrast():
    # Детерминированный баланс назначений для каждого типа потенциального исхода.
    n=30000
    a=np.tile([0,1,2],n//3)
    potential=np.tile(np.repeat(np.arange(100)/100,3),n//300)
    y=(potential < np.array([.1,.2,.3])[a]).astype(int)
    scores=arm_scores(y,a,np.full((n,3),.15))
    w=policy_weights(np.zeros(n),.2,2,random=True)
    gain=policy_scores(w,scores)-scores[:,0]
    assert np.isclose(gain.mean(),.04,atol=1e-12)

def test_budget_and_stochastic_baseline_equal_contacts():
    ranking=np.arange(103)
    actions=np.where(ranking%2,1,2)
    w=policy_weights(ranking,.2,actions)
    random=policy_weights(ranking,.2,1,random=True)
    assert np.isclose(w[:,1:].sum(),20)
    assert np.isclose(random[:,1:].sum(),20)
    assert np.allclose(w.sum(axis=1),1)

def test_invalid_policy_budget_rejected():
    with pytest.raises(ValueError):
        policy_weights([1,2],1.1)

def test_holm_monotonic_adjustment():
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])

def test_qini_endpoint_matches_random_baseline():
    from src.core import uplift_curve
    curve,area=uplift_curve(np.arange(6),[0,1,0,1,0,1],[0,1,0,1,0,1])
    assert curve.iloc[-1].gain == curve.iloc[-1].random
    assert np.isfinite(area)

def test_post_treatment_columns_excluded():
    import pandas as pd
    frame=pd.DataFrame(dict(recency=[1],history=[100],mens=[1],womens=[0],
        newbie=[0],zip_code=["Urban"],channel=["Web"],conversion=[1],visit=[1],spend=[40],action=[2]))
    assert not set(features(frame).columns)&{"conversion","visit","spend","action","segment"}

