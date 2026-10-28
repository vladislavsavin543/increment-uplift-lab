"""Сборка графиков, отчёта для бизнеса и автономного дашборда."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.localization import ARM_LABELS, MODEL_LABELS, STRATEGY_LABELS

def fmt_ci(row, scale=10000):
    return f'{row["estimate"]*scale:.1f} [{row["low"]*scale:.1f}; {row["high"]*scale:.1f}]'

def build_reports(root, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","figure.facecolor":"#f7f6f2",
        "axes.facecolor":"#f7f6f2","axes.labelcolor":"#20242a","text.color":"#20242a",
        "savefig.facecolor":"#f7f6f2","font.size":10})
    out=root/"reports"
    figdir=out/"figures"
    figdir.mkdir(exist_ok=True)
    policies=pd.DataFrame(result["policies"])
    effects=pd.DataFrame(result["effects"])
    colors={"random_mens":"#777c89","response":"#4f72c4","rule":"#d38935",
            "uplift":"#9365c8","multiarm":"#17866f"}
    labels=STRATEGY_LABELS
    fig,ax=plt.subplots(figsize=(10,5.8))
    for name,g in policies.groupby("strategy",sort=False):
        ax.plot(g.coverage*100,g.estimate*10000,label=labels[name],color=colors[name],lw=2.2)
    ax.axhline(0,color="#b5b5b5",lw=.8)
    ax.axvline(20,color="#b5b5b5",lw=.8,ls="--")
    ax.set(xlabel="Охват аудитории, %",ylabel="Дополнительные покупки на 10 000 клиентов",
           title="Больший охват не гарантирует лучшую стратегию")
    ax.legend(frameon=False)
    fig.text(.1,.01,"Отдельная тестовая выборка; оценка AIPW. Главное сравнение — при охвате 20%.",fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1])
    fig.savefig(figdir/"policy_curves.png",dpi=180)
    plt.close(fig)
    p20=policies[np.isclose(policies.coverage,.2)].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(10,4.6))
    for i,row in p20.iterrows():
        ax.errorbar(row.estimate*10000,i,xerr=[[max(0,(row.estimate-row.low)*10000)],
            [max(0,(row.high-row.estimate)*10000)]],fmt="o",capsize=4,color=colors[row.strategy])
    ax.set_yticks(range(len(p20)),[labels[x] for x in p20.strategy])
    ax.axvline(0,color="#777",lw=.8)
    ax.set(xlabel="Дополнительные покупки на 10 000 клиентов (индивидуальный 95% ДИ)",
           title="Одинаковый бюджет: охват 20% клиентов")
    fig.tight_layout()
    fig.savefig(figdir/"policy_20pct.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.8))
    conv=effects[effects.outcome=="conversion"].reset_index(drop=True)
    for i,row in conv.iterrows():
        ax.errorbar(row.estimate*100,i,xerr=[[(row.estimate-row.low)*100],
            [(row.high-row.estimate)*100]],fmt="o",capsize=5,color=["#9365c8","#17866f"][i])
    ax.set_yticks([0,1],["Письмо с мужскими товарами против контроля","Письмо с женскими товарами против контроля"])
    ax.axvline(0,color="#777",lw=.8)
    ax.set(xlabel="Эффект на конверсию, п.п. (95% ДИ без поправки)",
           title="Средние эффекты писем в рандомизированном эксперименте")
    fig.tight_layout()
    fig.savefig(figdir/"experiment_effects.png",dpi=180)
    plt.close(fig)
    curves=pd.DataFrame(result["curves"])
    fig,ax=plt.subplots(figsize=(9,5))
    for name,g in curves.groupby("strategy",sort=False):
        ax.plot(g.coverage*100,g.gain*10000,label=f'{labels[name]} (Qini={result["qini"][name]:.5f})',color=colors[name],lw=2)
    g=curves[curves.strategy=="uplift"]
    ax.plot(g.coverage*100,g.random*10000,color="#777c89",ls="--",label="Случайное ранжирование")
    ax.set(xlabel="Охват аудитории, %",ylabel="Дополнительные покупки по IPW на 10 000 клиентов",
        title="Проверка ранжирования: письмо с мужскими товарами против контроля")
    ax.legend(frameon=False,fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir/"qini.png",dpi=180)
    plt.close(fig)
    primary=result["primary"]
    verdict=("На тесте преимущество персонализации при охвате 20% подтверждается этим интервалом."
        if primary["low"]>0 else
        "На тесте убедительного преимущества персонализации при охвате 20% не установлено: интервал включает ноль."
        if primary["high"]>=0 else
        "На тесте персонализация при охвате 20% уступила случайной рассылке письма с мужскими товарами.")
    rows="\n".join(f'| {r["label"]} | {fmt_ci(r)} |' for r in p20.to_dict("records"))
    effectrows="\n".join(f'| {ARM_LABELS[r["arm"]]} | {r["control_mean"]*100:.3f}% | {r["treatment_mean"]*100:.3f}% | {r["estimate"]*100:+.3f} п.п. | {r["pvalue_holm"]:.4g} |'
        for r in conv.to_dict("records"))
    report=f"""# INCREMENT — кому отправлять письмо
## Решение для бизнеса
{verdict}
Главное сравнение: выбор письма аплифт-моделью против случайной рассылки с мужским ассортиментом, при одинаковом охвате 20%.
Разница на 10 000 клиентов: **{fmt_ci(primary)} дополнительных покупок**, 95% доверительный интервал.
Это оценка для зафиксированной стратегии в историческом эксперименте, а не обещание результата новой кампании.

## Что показал эксперимент
| Письмо | Контроль: покупки | Письмо: покупки | Средний эффект | p с поправкой Холма |
|---|---:|---:|---:|---:|
{effectrows}
Поправка Холма применяется к шести сравнениям: 2 письма × 3 исхода.
Интервалы на графике и в CSV — индивидуальные 95%, без поправки на множественность.
Покупки — основной исход, посещения и выручка — вторичные.
Не нужно подменять покупку посещением ради более красивого результата.

## Сравнение при одинаковом бюджете
Оценка дополнительных покупок относительно «никому не писать», на 10 000 клиентов.
| Стратегия | Оценка [95% ДИ] |
|---|---:|
{rows}
Сравнивать значимость разницы по пересечению этих отдельных интервалов некорректно.
Для сравнения стратегий используются парные разности AIPW-оценок:
[таблица сравнений](tables/test_paired_comparisons.csv).

## Рекомендация
Сначала оценить пользу самой рассылки; затем проверить, окупается ли персонализация.
Если преимущество модели не подтверждено, сохранить простую стратегию и проверить новую на свежем рандомизированном тесте.
Калькулятор в [dashboard.html](dashboard.html) показывает сценарии стоимости и маржи.
При выборе будущей стратегии зафиксировать её до нового эксперимента.

## Данные и дизайн
- Кевин Хиллстром / MineThatData, 2008; 64 000 клиентов; три случайно назначенные группы.
- Окно результата — две недели. Названия писем описывают ассортимент, а не пол клиента.
- CSV получен с зеркала scikit-uplift; MD5 проверен. SHA-256: data/raw/manifest.json.
- Пропуски: {result["quality"]["missing"]}; совпадающие полные профили: {result["quality"]["duplicate_profiles"]}.
  Совпадающие строки сохранены: уникального customer_id нет, одинаковые профили могут принадлежать разным людям.
- SRM: p={result["quality"]["srm_pvalue"]:.4f}; максимальный |SMD|={result["quality"]["max_abs_smd"]:.4f}.
  Проверки баланса не доказывают рандомизацию, но помогают обнаружить грубые проблемы.

## Как исключена утечка
Разбиение 60/20/20, начальное значение генератора 42; для второго разбиения — 43. Стратификация только по назначенному письму.
Обучение: {result["quality"]["split_counts"]["train"]}; валидация: {result["quality"]["split_counts"]["validation"]}; тест: {result["quality"]["split_counts"]["test"]}.
Покупок на тесте всего {result["quality"]["split_purchases"]["test"]}; оценки отдельных сегментов шумные.
Признаки: давность покупки, log1p прошлых расходов, прежние категории, новый клиент, тип локации, канал.
visit, conversion, spend и назначенная группа исключены из признаков.

Три кандидата T-learner: логистическая регрессия C=0.1, C=1 и неглубокий градиентный бустинг.
Модель для письма с мужскими товарами: {MODEL_LABELS[result["selection"]["selected_models"]["uplift"]]}.
Модель выбора письма: {MODEL_LABELS[result["selection"]["selected_models"]["multiarm"]]}.
Выбор — по AIPW-эффекту при 20% охвате на валидации.
Решения записаны в selection_lock.json до расчёта тестовых исходов.
Затем модели переобучены на объединении обучения и валидации; тест не участвует в обучении.
Вспомогательные модели исходов для AIPW — отдельные фиксированные логистические модели C=1.

## Как считается эффект стратегии
Для каждой группы a вычисляем:
    score_a = mu_a(X) + I(A=a) / (1/3) * (Y - mu_a(X))
Для стратегии с вероятностями действий pi_a:
    incremental_i = sum_a pi_a(X_i) * score_a - score_0
Среднее incremental_i оценивает изменение покупок на одного клиента всей аудитории.
Для сравнения стратегий берём парную разность этих величин на тех же клиентах.
Известная вероятность назначения 1/3 используется напрямую.
IPW без моделей исходов приведён рядом в CSV как проверка чувствительности.

Контактный бюджет — floor(q*n). Случайная стратегия аналитически усреднена по случайному выбору аудитории.
Модель покупки ранжирует по P(Y=1 | Mens, X); аплифт — по разнице P(Y=1 | Mens, X)-P(Y=1 | Control, X).
Стратегия выбора письма выбирает письмо с большей предсказанной разницей и ранжирует по этой разнице.
При фиксированном бюджете стратегия заполняет его даже при отрицательном предсказании.
Охват 0% позволяет отказаться от рассылки целиком.

Интервалы: среднее ± 1.96 * SE клиентских оценок. Они условны на обученной модели и
приближённо фиксированном пороге бюджета. Не включают вариацию повторного обучения и
оценки квантильного порога на новой аудитории; не являются одновременной полосой по всем охватам.
При редких покупках и малых охватах нормальное приближение может быть неточным.

## Qini
Диагностика только для письма с мужскими товарами против контроля.
После ограничения на эти две группы вероятность воздействия равна 1/2.
Кривая — накопленный дополнительный эффект по IPW. Qini — ненормированная площадь над случайной диагональю;
это не AUROC. Разные реализации Qini нельзя сравнивать без проверки определения.
Все охваты, кроме главного сравнения на 20%, — исследовательские результаты.

## Экономика и ограничения
Калькулятор: N * incremental_conversion * M - N * coverage * C,
где M — предполагаемая маржа на дополнительную покупку, C — стоимость контакта.
Это сценарий при одинаковой средней марже дополнительных покупок; фактическая прибыль не наблюдается.
Ни скидки, ни маржа, ни стоимость письма в данных не заданы.
Масштабирование на N предполагает такую же аудиторию и стабильный эффект.
Не учитываются будущие отписки, усталость от писем, повторные покупки и изменение ассортимента.
Индивидуальный эффект проверить невозможно: для клиента наблюдается только один исход.
Отрицательный прогноз не доказывает, что конкретному человеку письмо навредит.
Исторический результат 2008 года не переносится автоматически на современный бизнес.

## Материалы
- [Калькулятор и графики](dashboard.html)
- [Средние эффекты](tables/experiment_effects.csv)
- [Выбор модели](tables/model_selection.csv)
- [Тестовые стратегии](tables/test_policies.csv)
- [Баланс признаков](tables/balance.csv)
- [Исходное описание](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- [Зеркало и опубликованный MD5](https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html#fetch_hillstrom)
"""
    (out/"REPORT_RU.md").write_text(report,encoding="utf-8")
    template=(root/"src"/"dashboard.html").read_text(encoding="utf-8-sig")
    (out/"dashboard.html").write_text(template.replace("__RESULTS__",json.dumps(result,ensure_ascii=False).replace("</","<\\/")),encoding="utf-8")

