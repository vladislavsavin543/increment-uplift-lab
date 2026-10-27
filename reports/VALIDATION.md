# Validation record

- End-to-end analysis completed on Python 3.12.
- Raw compressed CSV MD5 matched the published scikit-uplift checksum.
- Schema, 64,000 rows, missingness, binary outcomes, finite monetary values and
  outcome consistency checked before model fitting.
- 10 pytest checks passed: known causal effect recovery, budget equality,
  zero-contact effect, valid action probabilities, Holm adjustment, Qini
  endpoint, exclusion of post-treatment predictors, disjoint complete splits,
  held-out prediction membership, independent primary contrast reconstruction.
- Notebook generated with executed computations and validated with nbformat.
- Matplotlib policy comparison visually inspected.
- Dashboard opened and visually inspected in the in-app browser.
- Browser checks confirmed zero contacts gives zero effect/cost, switching to
  random Mens targeting updates the comparison, scaling to 20,000 gives
  34.4 incremental purchases and 4,000 contacts, and negative contact cost
  triggers an explicit validation error.
- A separate headless Edge launch was unavailable in this environment.
  Functional browser verification used the supported in-app browser instead.
- A narrow viewport was not visually verified; responsive CSS is included.
- No live campaign, production deployment or new prospective experiment
  has been performed.

