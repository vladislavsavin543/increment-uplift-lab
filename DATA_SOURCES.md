# Data provenance and rights

Provider: Kevin Hillstrom / MineThatData.
Original experiment description:
https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

Public mirror used by the scikit-uplift Hillstrom loader:
https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz

Loader and published expected MD5:
https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html#fetch_hillstrom

Expected MD5: a68a81291f53a14f4e29002629803ba3.
The local download script also records SHA-256, size and URLs in data/raw/manifest.json.

The original article describes public analysis use. It does not establish an
explicit dataset license in the text reviewed for this project. Do not treat
the project's MIT code license as permission to relicense source data.
The shareable project ZIP omits raw and customer-level processed data.
Download from the public source using scripts/download_data.py before recomputing.

64,000 rows, 12 columns, no unique customer identifier.
Customer-level features cover the preceding 12 months; outcomes cover the
two weeks following email delivery. Three randomized groups.
The mirror preserves the category spelling "Surburban"; models treat it as a
categorical value. We do not silently correct or regroup it.

Raw fields:
- recency: months since prior purchase
- history_segment: binned previous-year spend (excluded from ML as redundant)
- history: previous-year spend, USD
- mens / womens: previous purchase in those departments, not customer gender
- newbie: new customer in the preceding year
- zip_code: Rural, Urban, Surburban classification; not a literal postal code
- channel: Phone, Web, Multichannel
- segment: randomized email assignment
- visit: visited in outcome window
- conversion: purchased in outcome window
- spend: observed spending in outcome window, USD

No campaign costs, margins, discounts, timestamps of individual purchases,
long-term opt-outs or future customer lifetime values are supplied.

