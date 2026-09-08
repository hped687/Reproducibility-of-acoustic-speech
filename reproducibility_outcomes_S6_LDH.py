
"""
====================================================================
SPEECH FEATURE REPRODUCIBILITY:
MEASUREMENT AGREEMENT + PREDICTION REPRODUCIBILITY
====================================================================

Purpose
-------
Quantify how much results can change when the SAME speech recordings
are analysed using different feature extraction implementations.

Analyses
--------
1. Extraction summary
2. Agreement between nominally equivalent features
3. Bland-Altman agreement
4. Psychiatric/depression prediction using each extraction method
5. Bootstrap 95% confidence intervals for prediction metrics
6. Feature-selection stability
7. Feature-selection overlap between extraction methods

INPUT
-----
results/feature_values_long.csv

AVEC 2017 outcome files:
    dev_split_Depression_AVEC2017.csv
    full_test_split.csv

OUTPUT
------
results/reproducibility/

    tables/
        S1_extraction_summary.csv
        S2_feature_agreement.csv
        S3_prediction_performance.csv
        S4_feature_selection.csv
        S5_feature_overlap.csv

    figures/
        Fig_S1_feature_agreement.png
        Fig_S2_BlandAltman_*.png
        Fig_S3_prediction_performance.png
        Fig_S4_feature_selection_overlap.png

    analysis_configuration.json

IMPORTANT
---------
The prediction analysis uses the SAME participants, SAME CV splits,
SAME feature-selection procedure and SAME classifier for every
extraction method.

Therefore, differences in predictive performance are attributable
to the feature extraction representation rather than differences
in modelling choices.

====================================================================
"""

from __future__ import annotations

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==================================================================
# CONFIGURATION
# ==================================================================

RESULTS_DIR = Path("") #insert path to data

FEATURE_FILE = (
    RESULTS_DIR /
    "feature_values_long.csv"
)

# --------------------------------------------------------------
# AVEC 2017 outcome files
# --------------------------------------------------------------

DEV_OUTCOME_FILE = Path(
    r"D:/Diac-woz/dev_split_Depression_AVEC2017.csv"
)

TEST_OUTCOME_FILE = Path(
    r"D:/Diac-woz/full_test_split.csv"
)

OUTCOME_PARTICIPANT_COL = "Participant_ID"

OUTCOME_COL = "PHQ_Binary"

PHQ_SCORE_COL = "PHQ_Score"

GENDER_COL = "Gender"


# ==================================================================
# PREDICTION SETTINGS
# ==================================================================

RANDOM_STATE = 42

N_REPEATS = 10

N_FOLDS = 5

N_BOOTSTRAP = 2000

TOP_K = 20


# ==================================================================
# OUTPUT DIRECTORIES
# ==================================================================

OUTPUT_DIR = RESULTS_DIR / "reproducibility"

TABLE_DIR = OUTPUT_DIR / "tables"

FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================================
# SKLEARN / SCIPY
# ==================================================================

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from scipy.stats import (
    pearsonr,
    spearmanr,
)


# ==================================================================
# LOAD FEATURE DATA
# ==================================================================

print("\n" + "=" * 80)
print("LOADING FEATURE DATA")
print("=" * 80)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"\nFeature file not found:\n{FEATURE_FILE.resolve()}\n\n"
        "Expected:\n"
        "results/feature_values_long.csv"
    )


features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature table:",
    features.shape
)

print(
    "\nFirst feature columns:"
)

print(
    features.columns.tolist()[:30]
)


# ==================================================================
# LOAD OUTCOME DATA
# ==================================================================

print("\n" + "=" * 80)
print("LOADING AVEC 2017 OUTCOME DATA")
print("=" * 80)


if not DEV_OUTCOME_FILE.exists():

    raise FileNotFoundError(
        f"\nDevelopment outcome file not found:\n"
        f"{DEV_OUTCOME_FILE.resolve()}"
    )


if not TEST_OUTCOME_FILE.exists():

    raise FileNotFoundError(
        f"\nTest outcome file not found:\n"
        f"{TEST_OUTCOME_FILE.resolve()}"
    )


dev_outcomes = pd.read_csv(
    DEV_OUTCOME_FILE
)

test_outcomes = pd.read_csv(
    TEST_OUTCOME_FILE
)

print(
    "Development outcomes:",
    dev_outcomes.shape
)

print(
    "Test outcomes:",
    test_outcomes.shape
)


# ==================================================================
# NORMALISE PARTICIPANT IDS
# ==================================================================

def extract_participant_id(value):
    """
    Extract numeric DAIC-WOZ participant ID.

    Examples
    --------
    300
    300.0
    "300"
    "300_P"
    "300_P.zip"
    "sub-300"
    "sub-300_P"
    """

    if pd.isna(value):

        return np.nan

    value = str(value)

    match = re.search(
        r"(?<!\d)(\d{3})(?!\d)",
        value,
    )

    if match:

        return match.group(1)

    return value.strip()


# ==================================================================
# PREPARE OUTCOME DATA
# ==================================================================

required_outcome_columns = [
    OUTCOME_PARTICIPANT_COL,
    OUTCOME_COL,
]


for col in required_outcome_columns:

    if col not in dev_outcomes.columns:

        raise ValueError(
            f"\nColumn '{col}' not found in development outcome file.\n"
            f"Available columns:\n{dev_outcomes.columns.tolist()}"
        )

    if col not in test_outcomes.columns:

        raise ValueError(
            f"\nColumn '{col}' not found in test outcome file.\n"
            f"Available columns:\n{test_outcomes.columns.tolist()}"
        )


# Add PHQ score if present

outcome_keep = [
    OUTCOME_PARTICIPANT_COL,
    OUTCOME_COL,
]

if PHQ_SCORE_COL in dev_outcomes.columns:

    outcome_keep.append(
        PHQ_SCORE_COL
    )

if GENDER_COL in dev_outcomes.columns:

    outcome_keep.append(
        GENDER_COL
    )


dev_outcomes = dev_outcomes[
    [
        c for c in outcome_keep
        if c in dev_outcomes.columns
    ]
].copy()


test_outcomes = test_outcomes[
    [
        c for c in outcome_keep
        if c in test_outcomes.columns
    ]
].copy()


# Combine

outcomes = pd.concat(
    [
        dev_outcomes,
        test_outcomes,
    ],
    ignore_index=True,
)


# Normalised participant ID

outcomes[
    "_participant_id"
] = outcomes[
    OUTCOME_PARTICIPANT_COL
].map(
    extract_participant_id
)


# Remove duplicated participant IDs.

# If an ID occurs in both files, keep the first occurrence.

outcomes = outcomes.drop_duplicates(
    subset="_participant_id",
    keep="first",
).copy()


print(
    "\nUnique outcome participants:",
    outcomes["_participant_id"].nunique()
)


# ==================================================================
# IDENTIFY FEATURE PARTICIPANT COLUMN
# ==================================================================

print("\n" + "=" * 80)
print("IDENTIFYING PARTICIPANT COLUMN")
print("=" * 80)


possible_id_columns = [

    "participant_id",

    "Participant_ID",

    "participant",

    "Participant",

    "participantID",

    "ParticipantID",

    "id",

    "ID",

    "zip_file",

    "wav_path_in_zip",

    "recording_id",

]


FEATURE_PARTICIPANT_COL = None


for col in possible_id_columns:

    if col in features.columns:

        FEATURE_PARTICIPANT_COL = col

        break


if FEATURE_PARTICIPANT_COL is None:

    raise ValueError(
        "\nCould not identify participant ID column "
        "in feature_values_long.csv.\n\n"
        "Available columns:\n"
        +
        "\n".join(
            features.columns.tolist()
        )
    )


print(
    "Using feature participant column:",
    FEATURE_PARTICIPANT_COL
)


# ==================================================================
# NORMALISE FEATURE PARTICIPANT IDS
# ==================================================================

features[
    "_participant_id"
] = features[
    FEATURE_PARTICIPANT_COL
].map(
    extract_participant_id
)


# ==================================================================
# MATCHING DIAGNOSTIC
# ==================================================================

feature_ids = set(
    features[
        "_participant_id"
    ].dropna()
)

outcome_ids = set(
    outcomes[
        "_participant_id"
    ].dropna()
)

matched_ids = (
    feature_ids &
    outcome_ids
)


print("\n" + "=" * 80)
print("PARTICIPANT ID MATCHING")
print("=" * 80)

print(
    "Unique feature participants:",
    len(feature_ids)
)

print(
    "Unique outcome participants:",
    len(outcome_ids)
)

print(
    "Matched participants:",
    len(matched_ids)
)

print(
    "Feature participants without outcome:",
    len(
        feature_ids -
        outcome_ids
    )
)

print(
    "Outcome participants without features:",
    len(
        outcome_ids -
        feature_ids
    )
)


if len(matched_ids) > 0:

    print(
        "\nExample matched IDs:",
        sorted(
            matched_ids
        )[:20]
    )


# ==================================================================
# MERGE
# ==================================================================

outcome_merge_columns = [
    "_participant_id",
    OUTCOME_COL,
]

if PHQ_SCORE_COL in outcomes.columns:

    outcome_merge_columns.append(
        PHQ_SCORE_COL
    )

if GENDER_COL in outcomes.columns:

    outcome_merge_columns.append(
        GENDER_COL
    )


data = features.merge(
    outcomes[
        outcome_merge_columns
    ],
    on="_participant_id",
    how="inner",
)


print(
    "\nMerged analysis data:",
    data.shape
)

print(
    "Matched participants:",
    data[
        "_participant_id"
    ].nunique()
)

print(
    "Matched recordings/rows:",
    len(data)
)


if data.empty:

    raise RuntimeError(
        "\n\nNO PARTICIPANTS MATCHED.\n\n"
        "Feature IDs and AVEC participant IDs do not overlap.\n\n"
        "First feature IDs:\n"
        +
        str(
            sorted(
                list(feature_ids)
            )[:20]
        )
        +
        "\n\nFirst outcome IDs:\n"
        +
        str(
            sorted(
                list(outcome_ids)
            )[:20]
        )
    )


# ==================================================================
# IDENTIFY FEATURE COLUMNS
# ==================================================================

print("\n" + "=" * 80)
print("IDENTIFYING FEATURE COLUMNS")
print("=" * 80)


method_prefixes = [
    "opensmile_",
    "praat_",
    "librosa_",
]


method_feature_sets = {}


for method in [
    "opensmile",
    "praat",
    "librosa",
]:

    cols = [
        c for c in data.columns
        if c.startswith(
            method + "_"
        )
    ]

    if len(cols) > 0:

        method_feature_sets[
            method
        ] = cols


print(
    "\nFeatures per method:"
)

for method, cols in method_feature_sets.items():

    print(
        f"  {method}: {len(cols)}"
    )


if len(method_feature_sets) < 2:

    raise RuntimeError(
        "\nFewer than two extraction methods were detected.\n"
        "Expected columns beginning with:\n"
        "  opensmile_\n"
        "  praat_\n"
        "  librosa_"
    )


# ==================================================================
# TABLE S1
# EXTRACTION SUMMARY
# ==================================================================

print("\nCreating Table S1...")

summary_rows = []


for method, cols in method_feature_sets.items():

    for feature in cols:

        values = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        summary_rows.append({

            "feature":
                feature,

            "method":
                method,

            "n":
                values.notna().sum(),

            "missing_pct":
                values.isna().mean() * 100,

            "mean":
                values.mean(),

            "sd":
                values.std(),

            "median":
                values.median(),

            "min":
                values.min(),

            "max":
                values.max(),

        })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    TABLE_DIR /
    "S1_extraction_summary.csv",
    index=False,
)


# ==================================================================
# AUTOMATICALLY IDENTIFY COMPARABLE FEATURES
# ==================================================================

"""
Rather than relying entirely on exact feature names, identify
nominally equivalent constructs.

This accommodates differences such as:

openSMILE:
    F0final
    F0semitoneFrom27.5Hz

Praat:
    f0_mean

librosa:
    f0_mean
"""


def find_first_column(
    columns,
    patterns,
):

    for pattern in patterns:

        for col in columns:

            if re.search(
                pattern,
                col,
                flags=re.IGNORECASE,
            ):

                return col

    return None


all_columns = data.columns.tolist()


# --------------------------------------------------------------
# F0
# --------------------------------------------------------------

opensmile_f0 = find_first_column(
    all_columns,
    [
        r"opensmile_.*f0.*amean",
        r"opensmile_.*f0",
    ],
)

praat_f0 = find_first_column(
    all_columns,
    [
        r"praat_.*f0.*mean",
        r"praat_.*f0",
    ],
)

librosa_f0 = find_first_column(
    all_columns,
    [
        r"librosa_.*f0.*mean",
        r"librosa_.*f0",
    ],
)


# --------------------------------------------------------------
# Jitter
# --------------------------------------------------------------

opensmile_jitter = find_first_column(
    all_columns,
    [
        r"opensmile_.*jitter.*amean",
        r"opensmile_.*jitter",
    ],
)

praat_jitter = find_first_column(
    all_columns,
    [
        r"praat_.*jitter.*local",
        r"praat_.*jitter",
    ],
)


# --------------------------------------------------------------
# Shimmer
# --------------------------------------------------------------

opensmile_shimmer = find_first_column(
    all_columns,
    [
        r"opensmile_.*shimmer.*amean",
        r"opensmile_.*shimmer",
    ],
)

praat_shimmer = find_first_column(
    all_columns,
    [
        r"praat_.*shimmer.*local",
        r"praat_.*shimmer",
    ],
)


# --------------------------------------------------------------
# HNR
# --------------------------------------------------------------

opensmile_hnr = find_first_column(
    all_columns,
    [
        r"opensmile_.*hnr.*amean",
        r"opensmile_.*hnr",
    ],
)

praat_hnr = find_first_column(
    all_columns,
    [
        r"praat_.*hnr.*mean",
        r"praat_.*hnr",
    ],
)


# --------------------------------------------------------------
# Loudness / energy
# --------------------------------------------------------------

opensmile_loudness = find_first_column(
    all_columns,
    [
        r"opensmile_.*loudness.*amean",
        r"opensmile_.*loudness",
    ],
)

praat_loudness = find_first_column(
    all_columns,
    [
        r"praat_.*intensity.*mean",
        r"praat_.*intensity",
    ],
)

librosa_loudness = find_first_column(
    all_columns,
    [
        r"librosa_.*rms.*mean",
        r"librosa_.*rms",
    ],
)


# ==================================================================
# BUILD COMPARISON LIST
# ==================================================================

COMPARISONS = {}


def add_comparison(
    construct,
    feature_a,
    feature_b,
):

    if (
        feature_a is None
        or feature_b is None
    ):

        return

    if (
        feature_a not in data.columns
        or feature_b not in data.columns
    ):

        return

    COMPARISONS.setdefault(
        construct,
        []
    ).append(
        (
            feature_a,
            feature_b,
        )
    )


# F0

add_comparison(
    "F0",
    opensmile_f0,
    praat_f0,
)

add_comparison(
    "F0",
    opensmile_f0,
    librosa_f0,
)

add_comparison(
    "F0",
    praat_f0,
    librosa_f0,
)


# Jitter

add_comparison(
    "Jitter",
    opensmile_jitter,
    praat_jitter,
)


# Shimmer

add_comparison(
    "Shimmer",
    opensmile_shimmer,
    praat_shimmer,
)


# HNR

add_comparison(
    "HNR",
    opensmile_hnr,
    praat_hnr,
)


# Loudness

add_comparison(
    "Loudness",
    opensmile_loudness,
    praat_loudness,
)

add_comparison(
    "Loudness",
    opensmile_loudness,
    librosa_loudness,
)

add_comparison(
    "Loudness",
    praat_loudness,
    librosa_loudness,
)


# ==================================================================
# MFCC COMPARISONS
# ==================================================================

for i in range(
    1,
    14,
):

    opensmile_mfcc = find_first_column(
        all_columns,
        [
            rf"opensmile_.*mfcc{i}.*amean",
            rf"opensmile_.*mfcc{i}",
        ],
    )

    librosa_mfcc = find_first_column(
        all_columns,
        [
            rf"librosa_.*mfcc[_]?{i}.*mean",
            rf"librosa_.*mfcc[_]?{i}",
        ],
    )

    add_comparison(
        f"MFCC{i}",
        opensmile_mfcc,
        librosa_mfcc,
    )


print("\nNominally comparable constructs:")

for construct, comparisons in COMPARISONS.items():

    print(
        f"\n{construct}:"
    )

    for a, b in comparisons:

        print(
            f"  {a}  <->  {b}"
        )


# ==================================================================
# AGREEMENT STATISTICS
# ==================================================================

def agreement_stats(
    df,
    feature_a,
    feature_b,
):

    x = pd.to_numeric(
        df[feature_a],
        errors="coerce",
    ).values

    y = pd.to_numeric(
        df[feature_b],
        errors="coerce",
    ).values

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 20:

        return None

    pearson_r, pearson_p = pearsonr(
        x,
        y,
    )

    spearman_rho, spearman_p = spearmanr(
        x,
        y,
    )

    difference = x - y

    mean_difference = np.mean(
        difference
    )

    mean_abs_difference = np.mean(
        np.abs(difference)
    )

    median_abs_difference = np.median(
        np.abs(difference)
    )

    sd_difference = np.std(
        difference,
        ddof=1,
    )

    loa_lower = (
        mean_difference
        -
        1.96 * sd_difference
    )

    loa_upper = (
        mean_difference
        +
        1.96 * sd_difference
    )

    pooled_sd = np.sqrt(
        (
            np.var(
                x,
                ddof=1,
            )
            +
            np.var(
                y,
                ddof=1,
            )
        )
        /
        2
    )

    if pooled_sd > 0:

        standardised_mae = (
            mean_abs_difference
            /
            pooled_sd
        )

    else:

        standardised_mae = np.nan


    # Relative difference.

    denominator = (
        np.abs(
            (x + y) / 2
        )
    )

    relative_difference = np.where(
        denominator > 1e-12,
        np.abs(x - y) / denominator,
        np.nan,
    )

    median_relative_difference = (
        np.nanmedian(
            relative_difference
        )
    )


    # Regression slope.

    if np.var(x) > 0:

        regression_slope = (
            np.cov(
                x,
                y,
                ddof=1,
            )[0, 1]
            /
            np.var(
                x,
                ddof=1,
            )
        )

        regression_intercept = (
            np.mean(y)
            -
            regression_slope *
            np.mean(x)
        )

    else:

        regression_slope = np.nan

        regression_intercept = np.nan


    return {

        "n":
            len(x),

        "pearson_r":
            pearson_r,

        "pearson_p":
            pearson_p,

        "spearman_rho":
            spearman_rho,

        "spearman_p":
            spearman_p,

        "mean_a":
            np.mean(x),

        "mean_b":
            np.mean(y),

        "sd_a":
            np.std(
                x,
                ddof=1,
            ),

        "sd_b":
            np.std(
                y,
                ddof=1,
            ),

        "mean_difference":
            mean_difference,

        "mean_absolute_difference":
            mean_abs_difference,

        "median_absolute_difference":
            median_abs_difference,

        "median_relative_difference":
            median_relative_difference,

        "standardised_mae":
            standardised_mae,

        "loa_lower":
            loa_lower,

        "loa_upper":
            loa_upper,

        "regression_slope":
            regression_slope,

        "regression_intercept":
            regression_intercept,

    }


# ==================================================================
# RUN AGREEMENT ANALYSIS
# ==================================================================

print("\n" + "=" * 80)
print("FEATURE AGREEMENT ANALYSIS")
print("=" * 80)


agreement_rows = []


for construct, comparisons in COMPARISONS.items():

    for feature_a, feature_b in comparisons:

        result = agreement_stats(
            data,
            feature_a,
            feature_b,
        )

        if result is None:

            continue

        result[
            "construct"
        ] = construct

        result[
            "feature_a"
        ] = feature_a

        result[
            "feature_b"
        ] = feature_b

        result[
            "method_a"
        ] = feature_a.split("_")[0]

        result[
            "method_b"
        ] = feature_b.split("_")[0]

        agreement_rows.append(
            result
        )


agreement = pd.DataFrame(
    agreement_rows
)


agreement_columns = [

    "construct",

    "method_a",

    "method_b",

    "feature_a",

    "feature_b",

    "n",

    "pearson_r",

    "pearson_p",

    "spearman_rho",

    "spearman_p",

    "mean_a",

    "mean_b",

    "sd_a",

    "sd_b",

    "mean_difference",

    "mean_absolute_difference",

    "median_absolute_difference",

    "median_relative_difference",

    "standardised_mae",

    "loa_lower",

    "loa_upper",

    "regression_slope",

    "regression_intercept",

]


if len(agreement) > 0:

    agreement = agreement[
        agreement_columns
    ]


agreement.to_csv(
    TABLE_DIR /
    "S2_feature_agreement.csv",
    index=False,
)


print(
    "\nAgreement comparisons:",
    len(agreement)
)


# ==================================================================
# PARTICIPANT-LEVEL PREDICTION DATA
# ==================================================================

print("\n" + "=" * 80)
print("PREPARING PREDICTION DATA")
print("=" * 80)


# IMPORTANT:
# Group using the normalised participant ID.

numeric_feature_columns = []


for method, cols in method_feature_sets.items():

    numeric_feature_columns.extend(
        cols
    )


# Convert feature columns to numeric.

for col in numeric_feature_columns:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce",
    )


# ==================================================================
# AGGREGATE ACOUSTIC FEATURES TO PARTICIPANT LEVEL
# ==================================================================

# IMPORTANT:
# Aggregate ONLY acoustic feature columns.
# Do not include PHQ_Binary, PHQ_Score, Gender, or other outcome/
# participant variables in the recording-level aggregation.

prediction_data = (
    data[
        [
            "_participant_id",
            *numeric_feature_columns,
        ]
    ]
    .groupby(
        "_participant_id",
        as_index=False,
    )
    .mean()
)


# ==================================================================
# ADD PARTICIPANT-LEVEL OUTCOME
# ==================================================================

participant_outcomes = (
    data[
        [
            "_participant_id",
            OUTCOME_COL,
        ]
    ]
    .drop_duplicates(
        subset="_participant_id"
    )
)


prediction_data = prediction_data.merge(
    participant_outcomes,
    on="_participant_id",
    how="left",
)


print(
    "Participant-level prediction data:",
    prediction_data.shape
)

print(
    "Participants:",
    prediction_data[
        "_participant_id"
    ].nunique()
)

print(
    "\nPrediction data columns containing outcome:"
)

print(
    [
        c for c in prediction_data.columns
        if "PHQ" in c.upper()
    ]
)

print(
    "Participant-level prediction data:",
    prediction_data.shape
)

print(
    "Participants:",
    prediction_data[
        "_participant_id"
    ].nunique()
)


# ==================================================================
# OUTCOME
# ==================================================================

y_series = prediction_data[
    OUTCOME_COL
]

y_numeric = pd.to_numeric(
    y_series,
    errors="coerce",
)


valid_outcome = np.isfinite(
    y_numeric
)


prediction_data = prediction_data[
    valid_outcome
].reset_index(
    drop=True
)


y = y_numeric[
    valid_outcome
].astype(int).values


print(
    "\nOutcome distribution:"
)

print(
    pd.Series(y).value_counts().sort_index()
)


if len(np.unique(y)) != 2:

    raise RuntimeError(
        "\nPrediction outcome does not contain exactly two classes."
    )


if min(
    np.bincount(y)
) < N_FOLDS:

    raise RuntimeError(
        "\nThere are not enough participants in one outcome class "
        f"for {N_FOLDS}-fold stratified CV."
    )


# ==================================================================
# SAME CV SPLITS FOR EVERY METHOD
# ==================================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_FOLDS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)


splits = list(
    cv.split(
        np.zeros(
            len(y)
        ),
        y,
    )
)


print(
    "\nRepeated CV splits:",
    len(splits)
)


# ==================================================================
# BOOTSTRAP CI FOR MULTIPLE METRICS
# ==================================================================

def bootstrap_metric_ci(
    y_true,
    predictions,
    n_bootstrap=2000,
    random_state=42,
):

    """
    Participant-level bootstrap 95% CIs.

    predictions are continuous probabilities.

    Returns:
        observed metric
        lower CI
        upper CI
    """

    rng = np.random.default_rng(
        random_state
    )

    y_true = np.asarray(
        y_true
    )

    predictions = np.asarray(
        predictions
    )

    predicted_class = (
        predictions >= 0.5
    ).astype(int)


    def calculate_metrics(
        y_b,
        p_b,
    ):

        pred_b = (
            p_b >= 0.5
        ).astype(int)

        auc = roc_auc_score(
            y_b,
            p_b,
        )

        accuracy = accuracy_score(
            y_b,
            pred_b,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_b,
                pred_b,
            )
        )

        sensitivity = recall_score(
            y_b,
            pred_b,
            zero_division=0,
        )

        precision = precision_score(
            y_b,
            pred_b,
            zero_division=0,
        )

        f1 = f1_score(
            y_b,
            pred_b,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_b,
            pred_b,
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        return {

            "AUC":
                auc,

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,

            "precision":
                precision,

            "F1":
                f1,

        }


    observed = calculate_metrics(
        y_true,
        predictions,
    )


    bootstrap_results = {
        metric: []
        for metric in observed
    }


    n = len(y_true)


    for _ in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        y_b = y_true[
            indices
        ]

        p_b = predictions[
            indices
        ]

        # Need both classes for AUC.

        if len(
            np.unique(y_b)
        ) < 2:

            continue


        metrics_b = calculate_metrics(
            y_b,
            p_b,
        )


        for metric, value in metrics_b.items():

            if np.isfinite(value):

                bootstrap_results[
                    metric
                ].append(
                    value
                )


    output = {}


    for metric, observed_value in observed.items():

        values = np.asarray(
            bootstrap_results[
                metric
            ]
        )

        if len(values) == 0:

            lower = np.nan

            upper = np.nan

        else:

            lower, upper = np.percentile(
                values,
                [2.5, 97.5],
            )


        output[
            metric
        ] = {

            "estimate":
                observed_value,

            "lower":
                lower,

            "upper":
                upper,

        }


    return output


# ==================================================================
# PREDICTION FUNCTION
# ==================================================================

def run_prediction(
    X,
    y,
    feature_names,
):

    """
    Identical modelling pipeline for every extraction method.

    Feature selection occurs INSIDE every training fold.

    Repeated-CV predictions are averaged for each participant.

    This produces a participant-level out-of-fold prediction that
    can then be used for bootstrap confidence intervals.
    """


    n_participants = len(y)

    prediction_matrix = np.full(
        (
            n_participants,
            len(splits),
        ),
        np.nan,
    )


    selected_features = []


    for split_number, (
        train_idx,
        test_idx,
    ) in enumerate(splits):


        X_train = X[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ]


        usable_k = min(
            TOP_K,
            X_train.shape[1],
        )


        if usable_k < 1:

            continue


        model = Pipeline([

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",

                StandardScaler(),
            ),

            (
                "selector",

                SelectKBest(
                    score_func=f_classif,
                    k=usable_k,
                ),
            ),

            (
                "classifier",

                LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),

        ])


        model.fit(
            X_train,
            y_train,
        )


        probabilities = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )


        prediction_matrix[
            test_idx,
            split_number,
        ] = probabilities


        selector = (
            model.named_steps[
                "selector"
            ]
        )


        selected = np.asarray(
            feature_names
        )[
            selector.get_support()
        ]


        for feature in selected:

            selected_features.append({

                "split":
                    split_number,

                "feature":
                    feature,

            })


        del model

        gc.collect()


    # --------------------------------------------------------------
    # Average predictions across repeats.
    # --------------------------------------------------------------

    participant_predictions = (
        np.nanmean(
            prediction_matrix,
            axis=1,
        )
    )


    valid = np.isfinite(
        participant_predictions
    )


    y_valid = y[
        valid
    ]

    p_valid = participant_predictions[
        valid
    ]


    # --------------------------------------------------------------
    # Bootstrap CIs
    # --------------------------------------------------------------

    metrics = bootstrap_metric_ci(
        y_valid,
        p_valid,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )


    return {

        "n":
            len(y_valid),

        "metrics":
            metrics,

        "selected_features":
            selected_features,

        "participant_predictions":
            participant_predictions,

    }


# ==================================================================
# RUN PREDICTION
# ==================================================================

print("\n" + "=" * 80)
print("PREDICTION REPRODUCIBILITY")
print("=" * 80)


prediction_rows = []

selection_records = {}

prediction_objects = {}


for method, feature_names in (
    method_feature_sets.items()
):


    print(
        "\n" + "-" * 70
    )

    print(
        "METHOD:",
        method,
    )


    # --------------------------------------------------------------
    # Numeric matrix
    # --------------------------------------------------------------

    X = (
        prediction_data[
            feature_names
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .values
    )


    # Remove completely missing columns.

    usable = np.isfinite(
        X
    ).any(
        axis=0
    )


    X = X[
        :,
        usable
    ]


    usable_feature_names = list(
        np.asarray(
            feature_names
        )[usable]
    )


    print(
        "Usable features:",
        len(usable_feature_names)
    )


    if len(
        usable_feature_names
    ) == 0:

        print(
            "No usable features. Skipping."
        )

        continue


    result = run_prediction(
        X,
        y,
        usable_feature_names,
    )


    prediction_objects[
        method
    ] = result


    metrics = result[
        "metrics"
    ]


    prediction_rows.append({

        "method":
            method,

        "n":
            result["n"],

        "n_features":
            len(
                usable_feature_names
            ),

        "top_k":
            min(
                TOP_K,
                len(
                    usable_feature_names
                ),
            ),


        "AUC":
            metrics[
                "AUC"
            ][
                "estimate"
            ],

        "AUC_95CI_lower":
            metrics[
                "AUC"
            ][
                "lower"
            ],

        "AUC_95CI_upper":
            metrics[
                "AUC"
            ][
                "upper"
            ],


        "accuracy":
            metrics[
                "accuracy"
            ][
                "estimate"
            ],

        "accuracy_95CI_lower":
            metrics[
                "accuracy"
            ][
                "lower"
            ],

        "accuracy_95CI_upper":
            metrics[
                "accuracy"
            ][
                "upper"
            ],


        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ][
                "estimate"
            ],

        "balanced_accuracy_95CI_lower":
            metrics[
                "balanced_accuracy"
            ][
                "lower"
            ],

        "balanced_accuracy_95CI_upper":
            metrics[
                "balanced_accuracy"
            ][
                "upper"
            ],


        "sensitivity":
            metrics[
                "sensitivity"
            ][
                "estimate"
            ],

        "sensitivity_95CI_lower":
            metrics[
                "sensitivity"
            ][
                "lower"
            ],

        "sensitivity_95CI_upper":
            metrics[
                "sensitivity"
            ][
                "upper"
            ],


        "specificity":
            metrics[
                "specificity"
            ][
                "estimate"
            ],

        "specificity_95CI_lower":
            metrics[
                "specificity"
            ][
                "lower"
            ],

        "specificity_95CI_upper":
            metrics[
                "specificity"
            ][
                "upper"
            ],


        "precision":
            metrics[
                "precision"
            ][
                "estimate"
            ],

        "precision_95CI_lower":
            metrics[
                "precision"
            ][
                "lower"
            ],

        "precision_95CI_upper":
            metrics[
                "precision"
            ][
                "upper"
            ],


        "F1":
            metrics[
                "F1"
            ][
                "estimate"
            ],

        "F1_95CI_lower":
            metrics[
                "F1"
            ][
                "lower"
            ],

        "F1_95CI_upper":
            metrics[
                "F1"
            ][
                "upper"
            ],

    })


    selection_records[
        method
    ] = result[
        "selected_features"
    ]


    print(
        f"AUC = "
        f"{metrics['AUC']['estimate']:.3f} "
        f"("
        f"{metrics['AUC']['lower']:.3f}"
        f"–"
        f"{metrics['AUC']['upper']:.3f}"
        f")"
    )


    print(
        f"Accuracy = "
        f"{metrics['accuracy']['estimate']:.3f} "
        f"("
        f"{metrics['accuracy']['lower']:.3f}"
        f"–"
        f"{metrics['accuracy']['upper']:.3f}"
        f")"
    )


    print(
        f"Balanced accuracy = "
        f"{metrics['balanced_accuracy']['estimate']:.3f} "
        f"("
        f"{metrics['balanced_accuracy']['lower']:.3f}"
        f"–"
        f"{metrics['balanced_accuracy']['upper']:.3f}"
        f")"
    )


    del X

    gc.collect()


# ==================================================================
# TABLE S3
# ==================================================================

prediction_results = pd.DataFrame(
    prediction_rows
)


prediction_results.to_csv(
    TABLE_DIR /
    "S3_prediction_performance.csv",
    index=False,
)


# ==================================================================
# TABLE S4 — FEATURE SELECTION
# ==================================================================

selection_rows = []


total_splits = len(
    splits
)


for method, records in (
    selection_records.items()
):


    counts = {}


    for record in records:

        feature = record[
            "feature"
        ]

        counts[
            feature
        ] = (
            counts.get(
                feature,
                0,
            )
            +
            1
        )


    for feature, count in counts.items():

        selection_rows.append({

            "method":
                method,

            "feature":
                feature,

            "selection_count":
                count,

            "selection_frequency":
                count /
                total_splits,

        })


selection = pd.DataFrame(
    selection_rows
)


if not selection.empty:

    selection = selection.sort_values(
        [
            "method",
            "selection_frequency",
        ],
        ascending=[
            True,
            False,
        ],
    )


selection.to_csv(
    TABLE_DIR /
    "S4_feature_selection.csv",
    index=False,
)


# ==================================================================
# TABLE S5 — FEATURE SELECTION OVERLAP
# ==================================================================

def get_selected_features(
    method,
    threshold=0.50,
):

    if selection.empty:

        return set()


    subset = selection[
        (
            selection["method"]
            ==
            method
        )
        &
        (
            selection[
                "selection_frequency"
            ]
            >=
            threshold
        )
    ]


    return set(
        subset[
            "feature"
        ]
    )


methods = list(
    method_feature_sets.keys()
)


overlap_rows = []


for i in range(
    len(methods)
):

    for j in range(
        i + 1,
        len(methods),
    ):


        method_a = methods[i]

        method_b = methods[j]


        set_a = get_selected_features(
            method_a
        )

        set_b = get_selected_features(
            method_b
        )


        union = (
            set_a |
            set_b
        )

        intersection = (
            set_a &
            set_b
        )


        if len(union) > 0:

            jaccard = (
                len(intersection)
                /
                len(union)
            )

        else:

            jaccard = np.nan


        overlap_rows.append({

            "method_a":
                method_a,

            "method_b":
                method_b,

            "n_selected_a":
                len(set_a),

            "n_selected_b":
                len(set_b),

            "n_shared":
                len(intersection),

            "jaccard_similarity":
                jaccard,

        })


overlap = pd.DataFrame(
    overlap_rows
)


overlap.to_csv(
    TABLE_DIR /
    "S5_feature_overlap.csv",
    index=False,
)


# ==================================================================
# FIGURE S1 — FEATURE AGREEMENT
# ==================================================================

import matplotlib.pyplot as plt


print(
    "\nCreating figures..."
)


if not agreement.empty:

    plt.figure(
        figsize=(10, 7)
    )


    plot_data = agreement.sort_values(
        "spearman_rho"
    )


    labels = (
        plot_data[
            "construct"
        ]
        + "\n"
        +
        plot_data[
            "method_a"
        ]
        + " vs "
        +
        plot_data[
            "method_b"
        ]
    )


    plt.barh(
        np.arange(
            len(plot_data)
        ),
        plot_data[
            "spearman_rho"
        ],
    )


    plt.yticks(
        np.arange(
            len(plot_data)
        ),
        labels,
    )


    plt.axvline(
        0,
        linewidth=1,
    )


    plt.xlabel(
        "Spearman correlation"
    )


    plt.title(
        "Agreement between acoustic feature extraction implementations"
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR /
        "Fig_S1_feature_agreement.png",
        dpi=300,
    )


    plt.close()


# ==================================================================
# FIGURE S2 — BLAND-ALTMAN
# ==================================================================

representative = [
    "F0",
    "Jitter",
    "Shimmer",
    "HNR",
    "Loudness",
    "MFCC1",
]


for construct in representative:


    subset = agreement[
        agreement[
            "construct"
        ]
        ==
        construct
    ]


    if subset.empty:

        continue


    row = subset.iloc[0]


    feature_a = row[
        "feature_a"
    ]

    feature_b = row[
        "feature_b"
    ]


    x = pd.to_numeric(
        data[
            feature_a
        ],
        errors="coerce",
    )


    y2 = pd.to_numeric(
        data[
            feature_b
        ],
        errors="coerce",
    )


    valid = (
        np.isfinite(x)
        &
        np.isfinite(y2)
    )


    x = x[
        valid
    ].values

    y2 = y2[
        valid
    ].values


    if len(x) < 20:

        continue


    mean_values = (
        x + y2
    ) / 2


    differences = (
        x - y2
    )


    mean_difference = (
        np.mean(
            differences
        )
    )


    sd_difference = (
        np.std(
            differences,
            ddof=1,
        )
    )


    lower = (
        mean_difference
        -
        1.96 * sd_difference
    )


    upper = (
        mean_difference
        +
        1.96 * sd_difference
    )


    plt.figure(
        figsize=(7, 5)
    )


    plt.scatter(
        mean_values,
        differences,
        alpha=0.5,
        s=20,
    )


    plt.axhline(
        mean_difference,
        linewidth=1,
    )


    plt.axhline(
        lower,
        linestyle="--",
        linewidth=1,
    )


    plt.axhline(
        upper,
        linestyle="--",
        linewidth=1,
    )


    plt.xlabel(
        "Mean of two implementations"
    )


    plt.ylabel(
        "Difference between implementations"
    )


    plt.title(
        f"Bland–Altman: {construct}"
    )


    plt.tight_layout()


    safe_name = (
        construct
        .replace(
            " ",
            "_",
        )
    )


    plt.savefig(
        FIGURE_DIR /
        f"Fig_S2_BlandAltman_{safe_name}.png",
        dpi=300,
    )


    plt.close()


# ==================================================================
# FIGURE S3 — PREDICTION PERFORMANCE
# ==================================================================

if not prediction_results.empty:

    plt.figure(
        figsize=(9, 6)
    )


    pred_plot = (
        prediction_results
        .copy()
    )


    x_positions = np.arange(
        len(pred_plot)
    )


    means = pred_plot[
        "AUC"
    ].values


    lower = pred_plot[
        "AUC_95CI_lower"
    ].values


    upper = pred_plot[
        "AUC_95CI_upper"
    ].values


    yerr = np.vstack([

        means - lower,

        upper - means,

    ])


    plt.errorbar(
        x_positions,
        means,
        yerr=yerr,
        fmt="o",
        capsize=5,
    )


    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )


    plt.xticks(
        x_positions,
        pred_plot[
            "method"
        ],
    )


    plt.ylabel(
        "Out-of-fold ROC AUC"
    )


    plt.xlabel(
        "Feature extraction implementation"
    )


    plt.title(
        "Prediction performance under different feature extraction methods"
    )


    plt.ylim(
        0,
        1,
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR /
        "Fig_S3_prediction_performance.png",
        dpi=300,
    )


    plt.close()


# ==================================================================
# FIGURE S4 — FEATURE SELECTION OVERLAP
# ==================================================================

if not overlap.empty:

    plt.figure(
        figsize=(8, 5)
    )


    labels = (
        overlap[
            "method_a"
        ]
        + " vs "
        +
        overlap[
            "method_b"
        ]
    )


    plt.bar(
        np.arange(
            len(overlap)
        ),
        overlap[
            "jaccard_similarity"
        ],
    )


    plt.xticks(
        np.arange(
            len(overlap)
        ),
        labels,
        rotation=45,
        ha="right",
    )


    plt.ylabel(
        "Jaccard similarity"
    )


    plt.title(
        "Overlap of selected acoustic features"
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR /
        "Fig_S4_feature_selection_overlap.png",
        dpi=300,
    )


    plt.close()


# ==================================================================
# SAVE PARTICIPANT-LEVEL PREDICTIONS
# ==================================================================

prediction_export = pd.DataFrame({

    "participant_id":
        prediction_data[
            "_participant_id"
        ],

    "outcome":
        y,

})


for method, result in (
    prediction_objects.items()
):

    prediction_export[
        f"{method}_oof_probability"
    ] = result[
        "participant_predictions"
    ]


prediction_export.to_csv(
    TABLE_DIR /
    "participant_level_oof_predictions.csv",
    index=False,
)


# ==================================================================
# SAVE ANALYSIS CONFIGURATION
# ==================================================================

configuration = {

    "feature_file":
        str(FEATURE_FILE),

    "development_outcome_file":
        str(DEV_OUTCOME_FILE),

    "test_outcome_file":
        str(TEST_OUTCOME_FILE),

    "n_participants":
        int(
            len(
                prediction_data
            )
        ),

    "n_recordings":
        int(
            len(data)
        ),

    "n_repeats":
        N_REPEATS,

    "n_folds":
        N_FOLDS,

    "total_cv_splits":
        len(splits),

    "top_k":
        TOP_K,

    "n_bootstrap":
        N_BOOTSTRAP,

    "random_state":
        RANDOM_STATE,

    "model":
        "LogisticRegression",

    "feature_selection":
        "SelectKBest(f_classif)",

    "preprocessing":
        [
            "median imputation",
            "standard scaling",
        ],

    "prediction_strategy":
        "Repeated stratified K-fold out-of-fold predictions averaged across repeats",

    "confidence_intervals":
        "Participant-level nonparametric bootstrap 95% CI",

}


with open(
    OUTPUT_DIR /
    "analysis_configuration.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        configuration,
        f,
        indent=2,
    )


# ==================================================================
# FINAL SUMMARY
# ==================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)


print(
    "\nParticipants:",
    len(
        prediction_data
    )
)


print(
    "Recordings:",
    len(data)
)


print(
    "\nTables:"
)

print(
    "  S1_extraction_summary.csv"
)

print(
    "  S2_feature_agreement.csv"
)

print(
    "  S3_prediction_performance.csv"
)

print(
    "  S4_feature_selection.csv"
)

print(
    "  S5_feature_overlap.csv"
)

print(
    "  participant_level_oof_predictions.csv"
)


print(
    "\nFigures:"
)

print(
    "  Fig_S1_feature_agreement.png"
)

print(
    "  Fig_S2_BlandAltman_*.png"
)

print(
    "  Fig_S3_prediction_performance.png"
)

print(
    "  Fig_S4_feature_selection_overlap.png"
)


print(
    "\nPrediction results:"
)


if not prediction_results.empty:

    display_columns = [

        "method",

        "n",

        "AUC",

        "AUC_95CI_lower",

        "AUC_95CI_upper",

        "accuracy",

        "accuracy_95CI_lower",

        "accuracy_95CI_upper",

        "balanced_accuracy",

        "balanced_accuracy_95CI_lower",

        "balanced_accuracy_95CI_upper",

        "sensitivity",

        "specificity",

        "F1",

    ]


    print(
        prediction_results[
            display_columns
        ].to_string(
            index=False
        )
    )


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR.resolve()
)


print(
    "\n" + "=" * 80
)

