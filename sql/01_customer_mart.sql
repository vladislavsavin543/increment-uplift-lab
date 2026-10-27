-- One row per randomized customer. source_row is a CSV row, not a customer ID.
CREATE OR REPLACE TABLE customer_mart AS
SELECT source_row, recency::INTEGER AS recency, history::DOUBLE AS history,
       history_segment, mens::INTEGER AS mens, womens::INTEGER AS womens,
       newbie::INTEGER AS newbie, zip_code, channel, segment,
       CASE segment WHEN 'No E-Mail' THEN 0 WHEN 'Mens E-Mail' THEN 1
                    WHEN 'Womens E-Mail' THEN 2 END AS action,
       visit::INTEGER AS visit, conversion::INTEGER AS conversion, spend::DOUBLE AS spend
FROM raw;

