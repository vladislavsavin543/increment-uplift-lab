"""Build a compact, executed walkthrough from reproducible analysis outputs."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    cells=[]
    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))
    env={"ROOT":ROOT}
    def code(text):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(text,"<notebook>","exec"),env)
        cell=nbformat.v4.new_code_cell(text)
        cell.execution_count=sum(c.cell_type=="code" for c in cells)+1
        cell.outputs=[nbformat.v4.new_output("stream",name="stdout",text=output.getvalue())]
        cells.append(cell)
    md("# INCREMENT — Uplift Lab\n\nИсполненный разбор проекта: эксперимент → признаки → отложенная оценка → бизнес-решение.\nПолная пересборка: из корня проекта выполнить python run.py. Этот ноутбук читает полученные артефакты и независимо пересчитывает основной эффект.")
    code("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
# Works when launched from either project root or notebooks/.
ROOT = Path.cwd()
if not (ROOT / 'run.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
R = json.loads((ROOT / 'reports/results.json').read_text(encoding='utf-8'))
print('Rows:', R['quality']['rows'])
print('Split:', R['quality']['split_counts'])
print('Versions:', R['versions'])""")
    md("## 1. О чём спрашиваем бизнес\nКому отправить мужское или женское письмо при ограниченном охвате? Основной исход — покупка за две недели. Главная проверка: персональный выбор письма против случайной рассылки мужского письма при охвате 20%.")
    code("""metrics = pd.read_csv(ROOT/'reports/tables/experiment_metrics.csv')
print(metrics.to_string(index=False))
effects = pd.read_csv(ROOT/'reports/tables/experiment_effects.csv')
print('\\nConversion contrasts:')
print(effects.query("outcome == 'conversion'")[['arm','estimate','low','high','pvalue_holm']].to_string(index=False))""")
    md("## 2. Разделяем эксперимент и ML\nНазначение писем случайное, вероятность 1/3. Разбиение 60/20/20 стратифицировано только по назначению. Модель выбрана на валидации, решения записаны до оценки теста. Совпадающие строки не удаляем: customer_id нет. Идентификатор source_row — лишь номер строки CSV.")
    code("""splits = pd.read_csv(ROOT/'data/processed/splits.csv')
print(splits.groupby('split').source_row.agg(['count','nunique']))
print('Unique across all splits:', splits.source_row.is_unique)
print('Selection lock:')
print(json.dumps(R['selection'],ensure_ascii=False,indent=2))""")
    md("## 3. Признаки и утечки\nИспользуем только историю до воздействия. Пол клиента неизвестен: mens/womens означают прежние покупки соответствующего ассортимента. visit, conversion и spend измерены после письма и исключены из X.")
    code("""from src.core import FEATURES
print('Pretreatment features:', FEATURES)
selection = pd.read_csv(ROOT/'reports/tables/model_selection.csv')
print(selection.to_string(index=False))""")
    md("## 4. Модель покупки и uplift\nT-learner обучает отдельную модель P(покупка | X, группа). Обычная response-стратегия ранжирует по прогнозу с мужским письмом; binary uplift — по разнице с контролем. Эти две стратегии используют одинаковые модели, различается именно ранжирование. Multi-arm выбирает письмо с наибольшей предсказанной разницей.")
    code("""h = pd.read_csv(ROOT/'data/processed/holdout_predictions.csv')
print(h[['binary_p_0','binary_p_1','multi_p_0','multi_p_1','multi_p_2']].head().round(5))
uplift = h[['multi_p_1','multi_p_2']].to_numpy()-h[['multi_p_0']].to_numpy()
print('Predicted uplift quantiles:', np.quantile(uplift.max(axis=1),[0,.25,.5,.75,1]))""")
    md("## 5. Оценка стратегии через AIPW\nДля каждого действия score_a = mu_a(X) + I(A=a)/e_a * (Y-mu_a(X)).\nЗатем сравниваем ожидаемый исход стратегии с действием «без письма». Модель mu обучена на train+validation, а не на тесте. Известная вероятность воздействия защищает оценку от неверной outcome-модели. Значения score не ограничиваем диапазоном [0,1].")
    code("""from src.core import arm_scores, policy_weights, policy_scores, summarize
mu = h[['mu_0','mu_1','mu_2']].to_numpy()
phi = arm_scores(h.conversion.to_numpy(),h.action.to_numpy(),mu)
score = uplift.max(axis=1)
best_email = uplift.argmax(axis=1)+1
personal = policy_weights(score,.2,best_email)
random = policy_weights(np.zeros(len(h)),.2,1,random=True)
personal_gain = policy_scores(personal,phi)-phi[:,0]
random_gain = policy_scores(random,phi)-phi[:,0]
comparison = summarize(personal_gain-random_gain,scale=10000)
print('Additional purchases / 10,000, personalized minus random:')
print(comparison)
print('Reproduces saved result:', np.isclose(comparison['estimate'],R['primary']['estimate']*10000))""")
    md("## 6. Что можно утверждать\nЕсли интервал парной разницы содержит ноль, преимущество этой персонализации не установлено. Это не доказывает равенство стратегий. Отдельные интервалы стратегий нельзя использовать вместо парного сравнения. Разные охваты на одном тесте — исследовательский анализ, не независимые подтверждения.")
    code("""policies = pd.read_csv(ROOT/'reports/tables/test_policies.csv')
p20 = policies[np.isclose(policies.coverage,.2)].copy()
for col in ['estimate','low','high']:
    p20[col] *= 10000
print(p20[['label','estimate','low','high']].to_string(index=False))""")
    img=base64.b64encode((ROOT/"reports/figures/policy_20pct.png").read_bytes()).decode()
    pic=nbformat.v4.new_code_cell("# Static figure generated by python run.py")
    pic.execution_count=sum(c.cell_type=="code" for c in cells)+1
    pic.outputs=[nbformat.v4.new_output("display_data",data={"image/png":img,"text/plain":"Equal-budget policy comparison"})]
    cells.append(pic)
    md("## 7. Экономика сценария\nДоп. прибыль = N × доп. вероятность покупки × маржа − N × охват × стоимость контакта. Маржа и стоимость придуманы для сценария, а не взяты из датасета. Интервалы условны на модели, не учитывают обучение и неопределённость переноса на новый бизнес.")
    code("""N, margin, cost, coverage = 10000, 25, .02, .2
gain = summarize(personal_gain)
expense = N*coverage*cost
print('Contacts:',int(N*coverage),'Cost:',expense)
print('Scenario incremental contribution:', N*gain['estimate']*margin-expense)
print('Scenario 95% CI:',(N*gain['low']*margin-expense,N*gain['high']*margin-expense))""")
    md("## 8. Следующий бизнес-шаг\nЗафиксировать сегментацию и охват до новой кампании, затем провести свежий рандомизированный тест стратегий. Для долгосрочного решения измерять покупки, маржу, отписки и повторные контакты. Подробные ограничения — reports/REPORT_RU.md; калькулятор — reports/dashboard.html.")
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
    nbformat.validate(notebook)
    nbformat.write(notebook,ROOT/"notebooks/01_uplift_walkthrough.ipynb")
    print("Executed notebook written")

if __name__=="__main__":
    # Notebook path discovery above uses cwd, so temporarily use project root.
    import os
    os.chdir(ROOT)
    main()

