-- Исследовательский анализ: эффекты сегментов на всех данных не используются для выбора стратегии.
WITH metrics AS (
    SELECT channel, action, COUNT(*) AS n, AVG(conversion) AS conversion_rate
    FROM customer_mart GROUP BY channel, action
), contrasts AS (
    SELECT t.channel, t.action, t.n AS treated_n, c.n AS control_n,
           t.conversion_rate-c.conversion_rate AS incremental_conversion
    FROM metrics t JOIN metrics c ON t.channel=c.channel AND c.action=0
    WHERE t.action>0
)
SELECT *, DENSE_RANK() OVER (PARTITION BY action ORDER BY incremental_conversion DESC) AS channel_rank
FROM contrasts ORDER BY action, channel_rank;

