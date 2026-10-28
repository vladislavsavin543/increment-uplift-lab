"""Сборка учебного ноутбука с русскими пояснениями и выполненными расчётами."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    env = {"ROOT": ROOT}

    def code(text):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(text, "<notebook>", "exec"), env)
        cell = nbformat.v4.new_code_cell(text)
        cell.execution_count = sum(c.cell_type == "code" for c in cells) + 1
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())]
        cells.append(cell)

    md("# INCREMENT — лаборатория аплифта\n\n"
       "Выполненный разбор: эксперимент → признаки → оценка на отдельном тесте → решение для бизнеса.\n\n"
       "Для полной пересборки выполни из корня проекта: **python run.py**. "
       "Этот ноутбук читает результаты и независимо пересчитывает главный эффект.")
    code("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
# Ноутбук запускается из корня проекта или из папки notebooks.
ROOT = Path.cwd()
if not (ROOT / 'run.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.localization import (
    ARM_LABELS, MODEL_LABELS, SPLIT_LABELS, STRATEGY_LABELS,
    FEATURE_LABELS, COLUMN_LABELS, table_for_display,
)
R = json.loads((ROOT / 'reports/results.json').read_text(encoding='utf-8'))
print('Клиентов:', R['quality']['rows'])
print('Разбиение:', {SPLIT_LABELS[k]: v for k, v in R['quality']['split_counts'].items()})
print('Версии библиотек:', R['versions'])""")
    md("## 1. Задача бизнеса\n"
       "Кому отправить письмо с мужским или женским ассортиментом при ограниченном охвате? "
       "Основной исход — покупка за две недели. Главная проверка: персональный выбор письма "
       "против случайной рассылки с мужским ассортиментом при охвате 20%.")
    code("""metrics = pd.read_csv(ROOT/'reports/tables/experiment_metrics.csv')
print(table_for_display(metrics).to_string(index=False))

effects = pd.read_csv(ROOT/'reports/tables/experiment_effects.csv')
print('\\nСредние эффекты на покупку:')
columns = ['arm', 'estimate', 'low', 'high', 'pvalue_holm']
print(table_for_display(effects.query("outcome == 'conversion'")[columns]).to_string(index=False))""")
    md("## 2. Разделяем эксперимент и машинное обучение\n"
       "Назначение писем случайное, вероятность каждой группы — 1/3. "
       "Разбиение 60/20/20 стратифицировано только по назначению. Модель выбирается на валидации; "
       "решения сохраняются до оценки теста. Совпадающие строки не удаляем: "
       "идентификатора клиента нет. source_row — лишь номер строки CSV.")
    code("""splits = pd.read_csv(ROOT/'data/processed/splits.csv')
print(table_for_display(splits.groupby('split').source_row.agg(['count', 'nunique'])))
print('Номера строк уникальны между всеми выборками:', bool(splits.source_row.is_unique))
lock = R['selection']
decisions = {
    'Начальное значение генератора': lock['seed'],
    'Основной исход': 'Покупка за две недели',
    'Основной охват': lock['primary_coverage'],
    'Выбранные модели': {
        STRATEGY_LABELS[k]: MODEL_LABELS[v] for k, v in lock['selected_models'].items()
    },
    'Критерий выбора': lock['selection_metric'],
    'Главное сравнение': lock['primary_contrast'],
    'Вероятности назначения': lock['propensity'],
    'SHA-256 разбиения': lock['splits_sha256'],
}
print('Зафиксированные решения:')
print(json.dumps(decisions, ensure_ascii=False, indent=2))""")
    md("## 3. Признаки и утечки\n"
       "Используем только историю до воздействия. Пол клиента неизвестен: "
       "mens/womens означают прежние покупки соответствующего ассортимента. "
       "visit, conversion и spend измерены после письма и исключены из признаков модели.")
    code("""from src.core import FEATURES
print('Признаки до рассылки:')
for feature in FEATURES:
    print(f'  {feature}: {FEATURE_LABELS[feature]}')
selection = pd.read_csv(ROOT/'reports/tables/model_selection.csv')
print('\\nСравнение кандидатов на валидации:')
print(table_for_display(selection).to_string(index=False))""")
    md("## 4. Модель покупки и аплифт\n"
       "T-learner обучает отдельную модель вероятности покупки для каждой группы. "
       "Стратегия вероятности покупки ранжирует клиентов по прогнозу с сообщением с мужским ассортиментом; "
       "аплифт письма с мужскими товарами — по разнице прогноза с письмом и без. "
       "Модели у этих двух стратегий одинаковые, отличается ранжирование. "
       "Стратегия выбора письма берёт вариант с наибольшим предсказанным приростом.")
    code("""h = pd.read_csv(ROOT/'data/processed/holdout_predictions.csv')
columns = ['binary_p_0', 'binary_p_1', 'multi_p_0', 'multi_p_1', 'multi_p_2']
print(table_for_display(h[columns].head().round(5)).to_string(index=False))
uplift = h[['multi_p_1', 'multi_p_2']].to_numpy()-h[['multi_p_0']].to_numpy()
quantiles = np.quantile(uplift.max(axis=1), [0, .25, .5, .75, 1])
print('Квантили предсказанного прироста (0%, 25%, 50%, 75%, 100%):', quantiles)""")
    md("## 5. Оценка стратегии через AIPW\n"
       "Для каждого действия: score_a = mu_a(X) + I(A=a)/e_a × (Y−mu_a(X)).\n\n"
       "Сравниваем ожидаемый исход стратегии с действием «без письма». "
       "Модель исходов mu обучена на объединении обучения и валидации. "
       "Известная вероятность воздействия позволяет корректировать ошибки модели исходов. "
       "Значения score не ограничиваем диапазоном [0,1].")
    code("""from src.core import arm_scores, policy_weights, policy_scores, summarize
mu = h[['mu_0', 'mu_1', 'mu_2']].to_numpy()
phi = arm_scores(h.conversion.to_numpy(), h.action.to_numpy(), mu)
score = uplift.max(axis=1)
best_email = uplift.argmax(axis=1)+1
personal = policy_weights(score, .2, best_email)
random = policy_weights(np.zeros(len(h)), .2, 1, random=True)
personal_gain = policy_scores(personal, phi)-phi[:, 0]
random_gain = policy_scores(random, phi)-phi[:, 0]
comparison = summarize(personal_gain-random_gain, scale=10000)
print('Дополнительные покупки на 10 000: персонализация минус случайный выбор')
for key, value in comparison.items():
    print(f'{COLUMN_LABELS[key]}: {value:.4f}')
matches = np.isclose(comparison['estimate'], R['primary']['estimate']*10000)
print('Сохранённый результат воспроизведён:', bool(matches))""")
    md("## 6. Что можно утверждать\n"
       "Если интервал парной разницы содержит ноль, преимущество персонализации не установлено. "
       "Это не доказывает равенство стратегий. Отдельные интервалы стратегий не заменяют "
       "парное сравнение. Другие охваты на том же тесте — исследовательский анализ.")
    code("""policies = pd.read_csv(ROOT/'reports/tables/test_policies.csv')
p20 = policies[np.isclose(policies.coverage, .2)].copy()
for column in ['estimate', 'low', 'high']:
    p20[column] *= 10000
print(table_for_display(p20[['label', 'estimate', 'low', 'high']]).to_string(index=False))""")
    img = base64.b64encode((ROOT/"reports/figures/policy_20pct.png").read_bytes()).decode()
    pic = nbformat.v4.new_code_cell("# График построен командой python run.py")
    pic.execution_count = sum(c.cell_type == "code" for c in cells)+1
    pic.outputs = [nbformat.v4.new_output("display_data", data={
        "image/png": img, "text/plain": "Стратегии при одинаковом бюджете: охват 20%",
    })]
    cells.append(pic)
    md("## 7. Экономика сценария\n"
       "Дополнительная прибыль = N × дополнительная вероятность покупки × маржа "
       "− N × охват × стоимость контакта. Маржа и стоимость — допущения сценария. "
       "Все суммы в долларах США; пересчёта в рубли нет. Интервалы условны на модели, "
       "не учитывают обучение и неопределённость переноса на новый бизнес.")
    code("""N, margin, cost, coverage = 10000, 25, .02, .2
gain = summarize(personal_gain)
expense = N*coverage*cost
print('Контактов:', int(N*coverage), 'Расходы, $:', expense)
print('Сценарная дополнительная прибыль, $:', N*gain['estimate']*margin-expense)
print('Сценарный 95% ДИ, $:', (
    N*gain['low']*margin-expense, N*gain['high']*margin-expense,
))""")
    md("## 8. Следующий шаг бизнеса\n"
       "Зафиксировать сегментацию и охват до новой кампании, затем провести новый "
       "рандомизированный тест стратегий. Для долгосрочного решения измерять покупки, "
       "маржу, отписки и повторные контакты. Ограничения — в reports/REPORT_RU.md; "
       "калькулятор — в reports/dashboard.html. Словарь исходных полей — в DATA_SOURCES.md; "
       "словарь выгружаемых таблиц — в reports/tables/README_RU.md.")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT/"notebooks/01_uplift_walkthrough.ipynb")
    print("Ноутбук с выполненными расчётами сохранён.")


if __name__ == "__main__":
    # Поиск корня в ячейках зависит от текущей папки.
    import os
    os.chdir(ROOT)
    main()
