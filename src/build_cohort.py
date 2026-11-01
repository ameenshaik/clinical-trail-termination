"""
CS 6140 Project - Predicting Early Termination of Clinical Trials
Step 1: Build the study cohort from AACT and print the numbers for the proposal.

What this script does
  1. Connects to the public AACT PostgreSQL database (free account required:
     https://aact.ctti-clinicaltrials.org).
  2. Selects interventional trials that started 2005-2019 and ended as
     COMPLETED (label 0) or TERMINATED (label 1).
  3. Prints the class counts ([N0], [N1]), the train/test split sizes and the
     snapshot (extraction) date -> paste these into the proposal.
  4. Joins registration-time features from several AACT tables and saves
     data/cohort.csv for the modeling steps.

Setup
  pip install pandas sqlalchemy psycopg2-binary
  export AACT_USER="your_username"
  export AACT_PASSWORD="your_password"
  python build_cohort.py            # structured features only
  python build_cohort.py --text     # also download brief-summary text
"""

import argparse
import os
import re
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text

AACT_HOST = "aact-db.ctti-clinicaltrials.org"
AACT_PORT = 5432
AACT_DB = "aact"

START_MIN = "2005-01-01"   # earliest start date kept
START_MAX = "2020-01-01"   # exclusive: trials started 2005-2019 (avoids right-censoring)
TRAIN_LAST_YEAR = 2015     # train/CV: 2005-2015, test: 2016-2019

# Cohort filter shared by every query. UPPER() makes it robust to how AACT
# stores enum values.
COHORT_SQL = f"""
    SELECT nct_id
    FROM studies
    WHERE UPPER(study_type) = 'INTERVENTIONAL'
      AND UPPER(overall_status) IN ('COMPLETED', 'TERMINATED')
      AND start_date >= '{START_MIN}'
      AND start_date <  '{START_MAX}'
"""

# Only columns known at registration. Deliberately NOT selected (label leakage):
#   why_stopped, completion_date, primary_completion_date, results_first_*,
#   last_update_*, and anything from the results/outcomes tables.
FEATURES_SQL = f"""
WITH cohort AS ({COHORT_SQL}),
sp AS (
    SELECT nct_id,
           MAX(agency_class) FILTER (WHERE LOWER(lead_or_collaborator) = 'lead') AS lead_sponsor_class,
           COUNT(*)          FILTER (WHERE LOWER(lead_or_collaborator) = 'collaborator') AS n_collaborators
    FROM sponsors WHERE nct_id IN (SELECT nct_id FROM cohort) GROUP BY nct_id
),
fac AS (
    SELECT nct_id, COUNT(*) AS n_sites
    FROM facilities WHERE nct_id IN (SELECT nct_id FROM cohort) GROUP BY nct_id
),
ctry AS (
    SELECT nct_id, COUNT(DISTINCT name) AS n_countries
    FROM countries WHERE nct_id IN (SELECT nct_id FROM cohort) GROUP BY nct_id
),
intv AS (
    SELECT nct_id, STRING_AGG(DISTINCT UPPER(intervention_type), '|') AS intervention_types,
           COUNT(*) AS n_interventions
    FROM interventions WHERE nct_id IN (SELECT nct_id FROM cohort) GROUP BY nct_id
),
cond AS (
    SELECT nct_id, COUNT(*) AS n_conditions
    FROM conditions WHERE nct_id IN (SELECT nct_id FROM cohort) GROUP BY nct_id
)
SELECT s.nct_id,
       UPPER(s.overall_status)          AS overall_status,
       s.phase,
       s.enrollment,
       UPPER(s.enrollment_type)         AS enrollment_type,
       s.start_date,
       s.number_of_arms,
       s.has_dmc,
       s.is_fda_regulated_drug,
       s.is_fda_regulated_device,
       d.allocation,
       d.intervention_model,
       d.primary_purpose,
       d.masking,
       e.gender,
       e.minimum_age,
       e.maximum_age,
       e.healthy_volunteers,
       LENGTH(e.criteria)               AS criteria_length,
       sp.lead_sponsor_class,
       COALESCE(sp.n_collaborators, 0)  AS n_collaborators,
       COALESCE(fac.n_sites, 0)         AS n_sites,
       COALESCE(ctry.n_countries, 0)    AS n_countries,
       intv.intervention_types,
       COALESCE(intv.n_interventions,0) AS n_interventions,
       COALESCE(cond.n_conditions, 0)   AS n_conditions
FROM studies s
JOIN cohort c            ON c.nct_id = s.nct_id
LEFT JOIN designs d      ON d.nct_id = s.nct_id
LEFT JOIN eligibilities e ON e.nct_id = s.nct_id
LEFT JOIN sp             ON sp.nct_id = s.nct_id
LEFT JOIN fac            ON fac.nct_id = s.nct_id
LEFT JOIN ctry           ON ctry.nct_id = s.nct_id
LEFT JOIN intv           ON intv.nct_id = s.nct_id
LEFT JOIN cond           ON cond.nct_id = s.nct_id
"""

TEXT_SQL = f"""
SELECT b.nct_id, b.description AS brief_summary
FROM brief_summaries b
WHERE b.nct_id IN ({COHORT_SQL})
"""


def get_engine():
    user, pwd = os.environ.get("AACT_USER"), os.environ.get("AACT_PASSWORD")
    if not user or not pwd:
        raise SystemExit("Set AACT_USER and AACT_PASSWORD environment variables first.")
    url = f"postgresql+psycopg2://{user}:{pwd}@{AACT_HOST}:{AACT_PORT}/{AACT_DB}"
    return create_engine(url)


def age_to_years(value):
    """Convert AACT age strings like '18 Years', '6 Months', 'N/A' to float years."""
    if value is None or pd.isna(value):
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)", str(value), re.I)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    per_year = {"year": 1, "month": 12, "week": 52.18, "day": 365.25,
                "hour": 8766, "minute": 525960}[unit]
    return n / per_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="also download brief-summary text")
    ap.add_argument("--out", default="data/cohort.csv")
    args = ap.parse_args()

    engine = get_engine()
    snapshot = date.today().isoformat()

    print("Querying AACT (this can take a minute)...")
    with engine.connect() as conn:
        df = pd.read_sql(text(FEATURES_SQL), conn)
        if args.text:
            txt = pd.read_sql(text(TEXT_SQL), conn)
            df = df.merge(txt, on="nct_id", how="left")

    # ---- label and split -------------------------------------------------
    df["label"] = (df["overall_status"] == "TERMINATED").astype(int)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["start_year"] = df["start_date"].dt.year
    df["split"] = df["start_year"].le(TRAIN_LAST_YEAR).map({True: "train", False: "test"})

    # ---- light cleaning (heavier preprocessing happens in the modeling step)
    df["min_age_years"] = df["minimum_age"].map(age_to_years)
    df["max_age_years"] = df["maximum_age"].map(age_to_years)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- numbers for the proposal ----------------------------------------
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    print("\n================ NUMBERS FOR THE PROPOSAL ================")
    print(f"Snapshot (extraction) date : {snapshot}")
    print(f"Total trials in cohort     : {len(df):,}")
    print(f"[N0] Completed (label 0)   : {n0:,}")
    print(f"[N1] Terminated (label 1)  : {n1:,}  ({n1 / len(df):.1%} of cohort)")

    print("\nTrain/test split by start year:")
    print(pd.crosstab(df["split"], df["overall_status"], margins=True).to_string())

    # Leakage check for the enrollment feature: completed trials usually report
    # ACTUAL enrollment, so look at this before deciding how to use it.
    print("\nEnrollment type by class (leakage check):")
    print(pd.crosstab(df["enrollment_type"].fillna("MISSING"),
                      df["overall_status"]).to_string())

    print("\nMissing values per column (top 10):")
    print(df.isna().mean().sort_values(ascending=False).head(10).map("{:.1%}".format).to_string())

    print(f"\nSaved {len(df):,} rows x {df.shape[1]} columns to {args.out}")
    print("\nSentence for the proposal:")
    print(f"  Expected sizes: Completed = {n0:,} trials and Terminated = {n1:,} trials "
          f"(AACT snapshot of {snapshot}).")


if __name__ == "__main__":
    main()