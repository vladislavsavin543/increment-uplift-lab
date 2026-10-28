"""Порядок действий: 0 — без письма, 1 — письмо с мужскими товарами, 2 — письмо с женскими товарами."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
NUMERIC = ["recency", "log_history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]
FEATURES = NUMERIC + CATEGORICAL
PROPENSITY = np.full(3, 1/3)

def features(frame):
    result = frame.copy()
    result["log_history"] = np.log1p(result["history"])
    return result[FEATURES]

def make_model(kind):
    transform = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    if kind.startswith("logit"):
        estimator = LogisticRegression(C=float(kind.split("_")[1]), max_iter=1000)
    elif kind == "boost":
        estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=200, l2_regularization=10, learning_rate=0.05,
            early_stopping=False, random_state=42)
    else:
        raise ValueError(f"Неизвестная модель: {kind}")
    return make_pipeline(transform, estimator)

def fit_tlearner(frame, kind):
    """Обучаем отдельные модели покупки на признаках до рассылки."""
    return [make_model(kind).fit(features(frame.loc[frame.action == a]),
        frame.loc[frame.action == a, "conversion"]) for a in range(3)]

def predict_arms(models, frame):
    x = features(frame)
    return np.column_stack([m.predict_proba(x)[:, 1] for m in models])

def arm_scores(y, action, mu, propensity=PROPENSITY):
    """AIPW: m_a(X) + 1[A=a]/e_a*(Y-m_a(X)); вероятность назначения e_a известна.
    Модели исходов обучаются отдельно от оцениваемых наблюдений. Оценки не обрезаем.
    """
    y, action, mu = np.asarray(y), np.asarray(action), np.asarray(mu, dtype=float)
    if mu.shape != (len(y), 3):
        raise ValueError("Массив mu должен иметь размерность (n, 3)")
    p = np.asarray(propensity)
    if p.shape != (3,) or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Некорректные вероятности назначения")
    scores = mu.copy()
    scores[np.arange(len(y)), action] += (y-mu[np.arange(len(y)), action])/p[action]
    return scores

def policy_weights(score, fraction, arm=1, random=False):
    """Бюджет — floor(q*n) контактов. Случайная стратегия усреднена по назначениям."""
    if not 0 <= fraction <= 1:
        raise ValueError("Доля контактов fraction должна быть в диапазоне [0, 1]")
    score = np.asarray(score)
    n = len(score)
    weights = np.zeros((n, 3))
    weights[:, 0] = 1
    k = int(np.floor(fraction*n))
    q = k/n if n else 0
    if random:
        weights[:, 0], weights[:, int(arm)] = 1-q, q
    else:
        chosen = np.argsort(-score, kind="stable")[:k]
        arms = np.full(n, arm, dtype=int) if np.isscalar(arm) else np.asarray(arm)
        if np.any(~np.isin(arms, [1, 2])):
            raise ValueError("Контактное действие должно быть 1 или 2")
        weights[chosen, 0] = 0
        weights[chosen, arms[chosen]] = 1
    return weights

def policy_scores(weights, scores):
    weights = np.asarray(weights)
    if weights.shape != scores.shape or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1):
        raise ValueError("Строки стратегии должны быть векторами вероятностей")
    return np.sum(weights*scores, axis=1)

def summarize(values, scale=1.0):
    """Индивидуальный нормальный 95% ДИ при зафиксированной обученной стратегии."""
    values = np.asarray(values, dtype=float)*scale
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    return dict(estimate=mean, se=se, low=mean-1.96*se, high=mean+1.96*se)

def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted

def experiment_effects(frame):
    rows = []
    for outcome in ["conversion", "visit", "spend"]:
        control = frame.loc[frame.action == 0, outcome].to_numpy()
        for arm in [1, 2]:
            treated = frame.loc[frame.action == arm, outcome].to_numpy()
            effect = float(treated.mean()-control.mean())
            se = np.sqrt(treated.var(ddof=1)/len(treated)+control.var(ddof=1)/len(control))
            rows.append(dict(outcome=outcome, action=arm, arm=ARMS[arm],
                control_mean=float(control.mean()), treatment_mean=float(treated.mean()),
                estimate=effect, low=effect-1.96*se, high=effect+1.96*se,
                pvalue=float(2*stats.norm.sf(abs(effect/se)))))
    result = pd.DataFrame(rows)
    result["pvalue_holm"] = holm(result.pvalue)
    return result

def uplift_curve(score, y, action):
    """Письмо с мужскими товарами против контроля, известная условная p=0,5. Ненормированный Qini.
    gain(q)=mean(1[rank<=q]*(A*Y/.5-(1-A)*Y/.5)).
    """
    mask = np.asarray(action) != 2
    s, y, a = np.asarray(score)[mask], np.asarray(y)[mask], np.asarray(action)[mask]
    order = np.argsort(-s, kind="stable")
    gain = np.r_[0, np.cumsum(np.where(a[order] == 1, 2*y[order], -2*y[order]))/len(y)]
    q = np.linspace(0, 1, len(y)+1)
    random_gain = q*gain[-1]
    qini = float(np.trapezoid(gain-random_gain, q))
    take = np.unique(np.linspace(0, len(y), 201).astype(int))
    return pd.DataFrame(dict(coverage=q[take], gain=gain[take], random=random_gain[take])), qini

