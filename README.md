# Predicting Early Termination of Clinical Trials

CS 6140 Machine Learning (Fall 2026), Northeastern University, Khoury College

**Team:** Ameen Shaik (002534243), Mohana Harshita Rangubhotla (002523567)

## Overview

We predict whether an interventional clinical trial will end as **Completed** or
**Terminated**, using only information available when the trial is registered
(phase, design, number of sites, sponsor type, intervention type, eligibility).
We compare logistic regression, SVM, a PyTorch deep neural network, and a random
forest, with hyperparameters tuned by stratified 5-fold cross-validation.

## Data

Source: [AACT](https://aact.ctti-clinicaltrials.org), a public PostgreSQL copy of
ClinicalTrials.gov. Snapshot used: **October 31, 2026**.

| | Completed (0) | Terminated (1) | Total |
|---|---|---|---|
| Train (start 2005–2015) | 109,752 | 14,222 | 123,974 |
| Test (start 2016–2019) | 55,896 | 7,165 | 63,061 |
| **All** | **165,648** | **21,387** | **187,035** |

The data is not stored in this repository. Rebuild it with the steps below.

## Setup

Requires Python 3.10+ and a free AACT account.

```bash
git clone <this-repo-url>
cd cs6140-clinical-trial-termination
pip install -r requirements.txt
```

Set your AACT credentials as environment variables (never commit them):

```bash
# macOS / Linux
export AACT_USER="your_username"
export AACT_PASSWORD="your_password"
```

```powershell
# Windows PowerShell
$env:AACT_USER="your_username"
$env:AACT_PASSWORD="your_password"
```

## Reproduce

```bash
# 1. Build the cohort -> data/cohort.csv (prints class counts)
python src/build_cohort.py
```

Further steps (preprocessing, model training, evaluation) will be added as the
project progresses.

## Repository structure

```
src/          Python scripts (data extraction, preprocessing, models)
notebooks/    Exploratory analysis
results/      Figures and metric tables
docs/         Proposal (LaTeX) and pitch slides
data/         Local data, ignored by git
```
