-- В знаменатели включаем ВСЕХ клиентов, случайно назначенных в группу.
SELECT segment, COUNT(*) AS customers, SUM(conversion) AS purchases,
       AVG(conversion) AS conversion_rate, AVG(visit) AS visit_rate,
       AVG(spend) AS revenue_per_customer, SUM(spend) AS total_revenue
FROM customer_mart GROUP BY segment ORDER BY segment;

