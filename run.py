"""Полная пересборка анализа. Запуск из любой папки: python run.py."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from src.core import (ARMS, features, fit_tlearner, predict_arms, arm_scores,
    policy_weights, policy_scores, summarize, experiment_effects, uplift_curve)
from scripts.download_data import main as download
from src.localization import STRATEGY_LABELS as LABELS

SEED = 42
COVERAGES = [0, .05, .1, .2, .3, .5, .75, 1]
PRIMARY_Q = .2
TABLES = ROOT/"reports"/"tables"

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def audit(raw):
    required = {"recency", "history_segment", "history", "mens", "womens", "zip_code",
                "newbie", "channel", "segment", "visit", "conversion", "spend"}
    if set(raw.columns) != required or len(raw) != 64000:
        raise ValueError("Неожиданная структура исходных данных или число строк")
    if raw.isna().any().any():
        raise ValueError("В исходных данных есть пропуски")
    if set(raw.segment.unique()) != set(ARMS):
        raise ValueError("Неожиданные названия экспериментальных групп")
    for name in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not raw[name].isin([0, 1]).all():
            raise ValueError(f"Поле {name} содержит значения, отличные от 0 и 1")
    if (raw[["history", "spend"]] < 0).any().any() or not np.isfinite(raw[["history", "spend"]]).all().all():
        raise ValueError("Некорректные денежные значения")
    if not raw.recency.between(1, 12).all():
        raise ValueError("Недопустимое число месяцев с прошлой покупки")
    if (raw.conversion > raw.visit).any() or ((raw.spend > 0) != raw.conversion.astype(bool)).any():
        raise ValueError("Исходы противоречат друг другу")
    counts = raw.segment.value_counts().reindex(ARMS)
    return dict(rows=len(raw), columns=len(raw.columns), missing=0,
        duplicate_profiles=int(raw.duplicated().sum()), duplicate_handling="Сохранены: идентификатора клиента нет; одинаковые профили могут принадлежать разным людям.",
        assignment_counts={k:int(v) for k,v in counts.items()},
        srm_pvalue=float(stats.chisquare(counts.to_numpy()).pvalue))

def balance(frame):
    x = features(frame)
    x = pd.get_dummies(x, columns=["zip_code", "channel"], dtype=float)
    rows = []
    for a in [1, 2]:
        for name in x.columns:
            treated = x.loc[frame.action == a, name].to_numpy(float)
            control = x.loc[frame.action == 0, name].to_numpy(float)
            denom = np.sqrt((treated.var(ddof=1)+control.var(ddof=1))/2)
            smd = (treated.mean()-control.mean())/denom if denom else 0
            rows.append(dict(action=a, feature=name, standardized_difference=float(smd)))
    return pd.DataFrame(rows)

def policy_inputs(frame, binary_predictions, multi_predictions):
    binary_gain = binary_predictions[:, 1]-binary_predictions[:, 0]
    multi_gain = multi_predictions[:, 1:]-multi_predictions[:, [0]]
    actions = multi_gain.argmax(axis=1)+1
    # Правило зафиксировано до оценки и не использует исходы.
    rule = -frame.recency.to_numpy()+np.log1p(frame.history.to_numpy())/100
    return {
        "random_mens": (np.zeros(len(frame)), 1, True),
        "response": (binary_predictions[:, 1], 1, False),
        "rule": (rule, 1, False),
        "uplift": (binary_gain, 1, False),
        "multiarm": (multi_gain.max(axis=1), actions, False),
    }

def evaluate(frame, binary_predictions, multi_predictions, nuisance):
    scores = arm_scores(frame.conversion.to_numpy(), frame.action.to_numpy(), nuisance)
    ipw = arm_scores(frame.conversion.to_numpy(), frame.action.to_numpy(), np.zeros_like(nuisance))
    inputs = policy_inputs(frame, binary_predictions, multi_predictions)
    rows, paired = [], []
    primary_vectors = {}
    for q in COVERAGES:
        vectors = {}
        for name, (ranking, actions, random) in inputs.items():
            weights = policy_weights(ranking, q, actions, random=random)
            gain = policy_scores(weights, scores)-scores[:, 0]
            ipw_gain = policy_scores(weights, ipw)-ipw[:, 0]
            vectors[name] = gain
            contacts = float(1-weights[:, 0].mean())
            rows.append(dict(strategy=name, label=LABELS[name], coverage=q,
                actual_coverage=contacts, n=len(frame), **summarize(gain),
                ipw_estimate=float(ipw_gain.mean()),
                mens_share=float(weights[:, 1].mean()), womens_share=float(weights[:, 2].mean())))
        for name in ["response", "rule", "uplift", "multiarm"]:
            paired.append(dict(strategy=name, coverage=q, comparator="random_mens",
                               **summarize(vectors[name]-vectors["random_mens"])))
        if q == PRIMARY_Q:
            primary_vectors = vectors
    # Главное сравнение зафиксировано при q=0,2; другие охваты — исследовательские.
    primary = summarize(primary_vectors["multiarm"]-primary_vectors["random_mens"])
    response_comparison = summarize(primary_vectors["multiarm"]-primary_vectors["response"])
    return pd.DataFrame(rows), pd.DataFrame(paired), primary, response_comparison

def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    download()
    raw = pd.read_csv(ROOT/"data"/"raw"/"hillstrom.csv.gz")
    quality = audit(raw)
    raw["source_row"] = np.arange(len(raw))
    con = duckdb.connect(str(ROOT/"data"/"processed"/"analytics.duckdb"))
    con.register("raw", raw)
    con.execute((ROOT/"sql"/"01_customer_mart.sql").read_text(encoding="utf-8-sig"))
    frame = con.sql("SELECT * FROM customer_mart ORDER BY source_row").df()
    # Разбиение стратифицировано только по группе; исходы не определяют выборку.
    train_idx, holdout_idx = train_test_split(np.arange(len(frame)), test_size=.4,
        random_state=SEED, stratify=frame.action)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=.5,
        random_state=SEED+1, stratify=frame.iloc[holdout_idx].action)
    membership = pd.Series("", index=frame.index)
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        membership.loc[idx] = name
    splits = pd.DataFrame({"source_row": frame.source_row, "split": membership})
    splits.to_csv(ROOT/"data"/"processed"/"splits.csv", index=False)
    train, val, test = [frame.iloc[idx].sort_values("source_row").copy()
                        for idx in [train_idx, val_idx, test_idx]]
    print(f"Разбиение: обучение={len(train)}, валидация={len(val)}, тест={len(test)}", flush=True)
    # Одна вспомогательная модель для сопоставимой оценки кандидатов на валидации.
    nuisance_train = fit_tlearner(train, "logit_1")
    val_mu = predict_arms(nuisance_train, val)
    val_scores = arm_scores(val.conversion.to_numpy(), val.action.to_numpy(), val_mu)
    candidates = ["logit_0.1", "logit_1", "boost"]
    selection = []
    val_predictions = {}
    for kind in candidates:
        print(f"Обучение кандидата: {kind}", flush=True)
        models = fit_tlearner(train, kind)
        pred = predict_arms(models, val)
        val_predictions[kind] = pred
        gains = pred[:, 1:]-pred[:, [0]]
        for mode in ["uplift", "multiarm"]:
            ranking = gains[:, 0] if mode == "uplift" else gains.max(axis=1)
            action = 1 if mode == "uplift" else gains.argmax(axis=1)+1
            w = policy_weights(ranking, PRIMARY_Q, action)
            result = summarize(policy_scores(w, val_scores)-val_scores[:, 0])
            selection.append(dict(model=kind, mode=mode, coverage=PRIMARY_Q, **result))
    selection = pd.DataFrame(selection)
    selected = {mode: str(selection.loc[selection[selection["mode"] == mode]["estimate"].idxmax(), "model"])
                for mode in ["uplift", "multiarm"]}
    # Сохраняем решения на диск ДО оценки тестовых исходов.
    lock = dict(seed=SEED, primary_outcome="conversion", primary_coverage=PRIMARY_Q,
        selected_models=selected, candidates=candidates, selection_metric="Дополнительные покупки по AIPW на валидации при охвате 20%",
        primary_contrast="Аплифт с выбором письма минус случайная рассылка мужского письма", propensity=[1/3]*3,
        splits_sha256=hashlib.sha256((ROOT/"data"/"processed"/"splits.csv").read_bytes()).hexdigest())
    save_json(ROOT/"reports"/"selection_lock.json", lock)
    selection.to_csv(TABLES/"model_selection.csv", index=False)
    val_table, _, _, _ = evaluate(val, val_predictions[selected["uplift"]], val_predictions[selected["multiarm"]], val_mu)
    val_table.to_csv(TABLES/"validation_policies.csv", index=False)
    fit_frame = pd.concat([train, val]).sort_values("source_row")
    fitted = {}
    for kind in set([*selected.values(), "logit_1"]):
        fitted[kind] = fit_tlearner(fit_frame, kind)
    pred_binary = predict_arms(fitted[selected["uplift"]], test)
    pred_multi = predict_arms(fitted[selected["multiarm"]], test)
    mu_test = predict_arms(fitted["logit_1"], test)
    test_table, paired, primary, vs_response = evaluate(test, pred_binary, pred_multi, mu_test)
    test_table.to_csv(TABLES/"test_policies.csv", index=False)
    paired.to_csv(TABLES/"test_paired_comparisons.csv", index=False)
    # Сохраняем тестовые прогнозы для независимой проверки и пересчёта.
    export = test[["source_row", "action", "conversion", "visit", "spend"]].copy()
    for a in range(3):
        export[f"mu_{a}"] = mu_test[:, a]
        export[f"binary_p_{a}"] = pred_binary[:, a]
        export[f"multi_p_{a}"] = pred_multi[:, a]
    export.to_csv(ROOT/"data"/"processed"/"holdout_predictions.csv", index=False)
    # Описательные итоги полного эксперимента считаем после фиксации выбора.
    quality.update(split_counts=membership.value_counts().to_dict(),
        split_purchases={name: int(frame.loc[membership == name, "conversion"].sum()) for name in ["train","validation","test"]})
    metrics = con.sql((ROOT/"sql"/"02_experiment_metrics.sql").read_text(encoding="utf-8-sig")).df()
    metrics.to_csv(TABLES/"experiment_metrics.csv", index=False)
    con.sql((ROOT/"sql"/"03_segments.sql").read_text(encoding="utf-8-sig")).df().to_csv(TABLES/"exploratory_channels.csv", index=False)
    con.close()
    effects = experiment_effects(frame)
    effects.to_csv(TABLES/"experiment_effects.csv", index=False)
    balance_table = balance(frame)
    balance_table.to_csv(TABLES/"balance.csv", index=False)
    quality["max_abs_smd"] = float(balance_table.standardized_difference.abs().max())
    save_json(ROOT/"reports"/"data_quality.json", quality)
    curves, qini = [], {}
    for name, score in [("response", pred_binary[:,1]), ("rule", policy_inputs(test,pred_binary,pred_multi)["rule"][0]),
                        ("uplift", pred_binary[:,1]-pred_binary[:,0])]:
        curve, area = uplift_curve(score, test.conversion.to_numpy(), test.action.to_numpy())
        curve["strategy"] = name
        curves.append(curve)
        qini[name] = area
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(TABLES/"uplift_curves.csv", index=False)
    summary = dict(quality=quality, selection=lock, primary=primary, vs_response=vs_response,
        qini=qini, arms=metrics.to_dict("records"), effects=effects.to_dict("records"),
        policies=test_table.to_dict("records"), validation=val_table.to_dict("records"),
        paired=paired.to_dict("records"), curves=curves.to_dict("records"),
        versions={name:importlib.metadata.version(name) for name in ["numpy","pandas","scikit-learn","scipy","duckdb","matplotlib"]})
    save_json(ROOT/"reports"/"results.json", summary)
    from src.reporting import build_reports
    build_reports(ROOT, summary)
    print("Главная разница на 10 000 клиентов:", {k:round(v*10000,2) for k,v in primary.items()}, flush=True)
    print("Готово: reports/dashboard.html и reports/REPORT_RU.md", flush=True)

if __name__ == "__main__":
    main()

