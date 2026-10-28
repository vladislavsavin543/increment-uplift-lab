"""Русские подписи для отчёта, графиков и учебного ноутбука.

Перевод применяется к представлению; исходные поля и идентификаторы не меняются.
"""

STRATEGY_LABELS = {
    "random_mens": "Случайно / письмо с мужскими товарами",
    "response": "Вероятность покупки",
    "rule": "Недавние покупки",
    "uplift": "Аплифт / письмо с мужскими товарами",
    "multiarm": "Аплифт / выбор письма",
}
ARM_LABELS = {
    "No E-Mail": "Без письма",
    "Mens E-Mail": "Письмо с мужскими товарами",
    "Womens E-Mail": "Письмо с женскими товарами",
}
MODEL_LABELS = {
    "logit_0.1": "Логистическая регрессия, C=0,1",
    "logit_1": "Логистическая регрессия, C=1",
    "boost": "Градиентный бустинг",
}
SPLIT_LABELS = {"train": "Обучение", "validation": "Валидация", "test": "Тест"}
FEATURE_LABELS = {
    "recency": "Месяцев с прошлой покупки",
    "log_history": "Логарифм прошлых расходов: log1p(history)",
    "mens": "Прежние покупки письма с мужскими товарами",
    "womens": "Прежние покупки письма с женскими товарами",
    "newbie": "Новый клиент",
    "zip_code": "Тип местности",
    "channel": "Канал прошлых покупок",
}
COLUMN_LABELS = {
    "segment": "Группа", "customers": "Клиентов", "purchases": "Покупок",
    "conversion_rate": "Конверсия", "visit_rate": "Доля посещений",
    "revenue_per_customer": "Выручка на клиента, $", "total_revenue": "Выручка, $",
    "arm": "Письмо", "outcome": "Исход", "estimate": "Оценка",
    "low": "Нижняя граница 95% ДИ", "high": "Верхняя граница 95% ДИ",
    "se": "Стандартная ошибка", "pvalue_holm": "p с поправкой Холма",
    "count": "Строк", "nunique": "Уникальных строк", "split": "Выборка",
    "model": "Модель", "mode": "Стратегия", "coverage": "Охват",
    "label": "Стратегия", "binary_p_0": "Без письма: модель письма с мужскими товарами",
    "binary_p_1": "Письмо с мужскими товарами: модель письма с мужскими товарами",
    "multi_p_0": "Без письма: модель выбора письма",
    "multi_p_1": "Письмо с мужскими товарами: модель выбора письма",
    "multi_p_2": "Письмо с женскими товарами: модель выбора письма",
}


def table_for_display(frame):
    """Возвращает копию таблицы с русскими подписями для чтения в ноутбуке."""
    values = {**ARM_LABELS, **STRATEGY_LABELS, **MODEL_LABELS, **SPLIT_LABELS}
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(lambda value: values.get(value, value))
    result = result.rename(columns=COLUMN_LABELS, index=SPLIT_LABELS)
    result.index.name = COLUMN_LABELS.get(result.index.name, result.index.name)
    return result
