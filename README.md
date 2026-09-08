# Reproducibility of acoustic speech feature extraction in psychiatry

Code and analysis materials accompanying the manuscript:

> **Reproducibility of acoustic speech feature extraction in psychiatry: a systematic review of methods, software and open-source repositories**

## Overview

This repository contains the Python scripts used to generate the tables, figures, supplementary materials, and illustrative feature-extraction reproducibility case study for a systematic review of acoustic speech-feature extraction in psychiatric research.

The repository has two related components:

1. **Systematic-review analysis.** Scripts harmonise the study-level extraction and repository-appraisal datasets; generate analysis-ready long-format datasets; produce descriptive tables, quality summaries, figures, QA/audit files, and a consolidated Excel workbook.
2. **Illustrative extraction reproducibility case study.** Scripts extract acoustic features from the same WAV recordings using openSMILE, Praat/Parselmouth, and librosa, quantify agreement among nominally comparable constructs, and compare downstream depression-prediction performance under a common modelling protocol.

The scripts intentionally keep extraction-method-specific feature names rather than treating features with similar labels as interchangeable. Agreement analyses therefore evaluate *nominally comparable* constructs; they do not establish mathematical equivalence among implementations.

## Repository contents

| File | Purpose |
|---|---|
| `analysis_consolidation_LDH-4.py` | Main systematic-review analysis pipeline. Cleans the study export; applies optional study and repository overrides; harmonises diagnoses, languages, tasks, acoustic feature families, software, and validation approaches; writes long-format datasets, tables, figures, QA files, an Excel summary workbook, a Markdown run summary, and a JSON analysis manifest. |
| `figures_LDH-3.py` | Standalone figure-generation script for the review’s Figure layouts. Produces diagnosis-by-feature/software tile matrices, reporting-completeness and repository-quality plots, and the corresponding plotted-value CSV files. |
| `reproducibility_casestudy_S5_LDH.py` | Extracts acoustic features from WAV files stored within ZIP archives using openSMILE eGeMAPSv02 functionals, Praat via Parselmouth, and librosa. Outputs recording-level features, descriptive summaries, cross-implementation agreement statistics, processing logs, and extraction-error records. |
| `reproducibility_outcomes_S5_LDH-2.py` | Analyses extraction agreement and downstream prediction reproducibility using the case-study feature table and AVEC 2017 outcome files. Produces supplementary tables, agreement and Bland–Altman figures, repeated cross-validated depression-prediction results, bootstrap confidence intervals, and feature-selection stability/overlap results. |

## Requirements

### Python

Use Python 3.10 or later. The scripts were written for a scientific Python environment and have not been packaged as an installable module.

Install the common dependencies with:

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn openpyxl
```

For the case-study feature extraction, also install:

```bash
python -m pip install opensmile praat-parselmouth librosa
```

`praat-parselmouth` provides Python access to Praat. If an extraction package is unavailable, `reproducibility_casestudy_S5_LDH.py` detects this at run time and skips that implementation; at least one of openSMILE, Parselmouth/Praat, or librosa must be available.

### Software versions

For a publication-grade reproduction, record the exact Python version and installed package versions used for your run. The systematic-review pipeline automatically records Python version, input-file SHA-256 hashes, run time, input paths, and output counts in `analysis_manifest.json`. The case-study outcome script writes its modelling settings to `analysis_configuration.json`.

An example environment setup is:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn openpyxl opensmile praat-parselmouth librosa
```

## Data availability and privacy

The repository does **not** include article-screening data, study-level extraction data, repository-appraisal workbooks, DAIC-WOZ/AVEC audio, participant-level outcomes, or generated results. These materials may be subject to copyright, database access conditions, or participant-data governance restrictions.

To reproduce the analyses:

- Obtain the review datasets through the study authors or the relevant project-controlled location.
- Obtain any AVEC/DAIC-WOZ-derived audio and outcome data through the dataset’s authorised access route and use them only under its applicable terms.
- Do not commit restricted audio, participant-level outcomes, identifiers, or derived files that could enable re-identification.

Paths in the scripts are local configuration placeholders and must be updated before execution.

## Systematic-review pipeline

### Inputs

`analysis_consolidation_LDH-4.py` requires two inputs:

| Argument | Required input | Required content |
|---|---|---|
| `--studies` | CSV export of the included publications | At minimum: `Covidence #`, `Study ID`, `Year`, `Diagnostic Group(s)`, `Software / tool / package`, and `Feature(s)` |
| `--github-workbook` | Quality-reviewed repository workbook (`.xlsx`) | Worksheets named `RepositoryMaster`, `QualityAppraisal`, and `StudyRepoLinks` |

The pipeline can additionally take:

| Argument | Optional input | Required key columns |
|---|---|---|
| `--repo-overrides` | CSV specifying repository-level analytic adjudications | `repo_key`, with optional `final_decision` and `notes` |
| `--study-overrides` | CSV specifying study-level inclusion/sensitivity adjudications | `covidence_number`, with optional `primary_include`, `sensitivity_group`, and `notes` |

Before running, inspect the input-column names and adjust the code or input files if your export differs materially from the expected schema. The script performs basic input validation and writes unmatched free-text values to the QA folder for manual review.

### Run

```bash
python analysis_consolidation_LDH-4.py \
  --studies review_publications.csv \
  --github-workbook github_repository_review_workbook.xlsx \
  --repo-overrides repository_analysis_overrides.csv \
  --study-overrides study_analysis_overrides.csv \
  --output-dir systematic_review_analysis_outputs
```

In Windows PowerShell, use backticks for line continuation, or place the command on one line:

```powershell
python .\analysis_consolidation_LDH-4.py --studies .\review_publications.csv --github-workbook .\github_repository_review_workbook.xlsx --output-dir .\systematic_review_analysis_outputs
```

### Outputs

The output directory contains:

```text
systematic_review_analysis_outputs/
├── long_format/
├── tables/
├── figures/
├── qa/
├── analysis_manifest.json
├── analysis_summary.md
└── systematic_review_analysis_summary.xlsx
```

Key output groups are:

- `long_format/`: cleaned study data and one-to-many, analysis-ready datasets for diagnoses, countries, languages, software, programming languages, feature families/sets, tasks, validation, and study–repository links.
- `tables/`: descriptive counts and percentages; reporting-completeness, open-science, repository-class, README-status, access-status, and repository-quality summaries; diagnosis-by-feature and diagnosis-by-software matrices.
- `figures/`: publication-year, diagnostic-group, feature-family, software, reporting, validation, cross-tabulation, and repository-quality plots.
- `qa/`: data repairs and unmatched terms requiring review. These files are an audit layer and should be inspected before final interpretation.
- `systematic_review_analysis_summary.xlsx`: a formatted workbook consolidating major tables and headline statistics.
- `analysis_manifest.json`: run provenance, including input checksums and warnings.

## Standalone review figures

`figures_LDH-3.py` creates alternative layouts for the review figures. Its default folder layout is:

```text
<PROJECT_DIR>/
├── Data/
│   ├── 28Jul26_FINAL_v2.csv
│   └── 28Jul26_FINAL_GITHUB.xlsx
└── outputs/
```

Set `PROJECT_DIR` near the top of the script, or modify the expected filenames if required. The script searches for a single unambiguous filename matching its fallback patterns when the preferred names are absent.

Run:

```bash
python figures_LDH-3.py --data-dir Data --output-dir outputs
```

Primary outputs are:

```text
outputs/
├── Figure_2_alternative_tile_matrices.png
├── Figure_2_alternative_tile_matrices.pdf
├── Figure_2_alternative_tile_matrices.svg
├── Figure_3_alternative_reproducibility.png
├── Figure_3_alternative_reproducibility.pdf
├── Figure_3_alternative_reproducibility.svg
├── Figure_2A_feature_counts.csv
├── Figure_2A_feature_percentages.csv
├── Figure_2B_software_counts.csv
├── Figure_2B_software_percentages.csv
├── Figure_3A_reporting_completeness.csv
└── Figure_3B_repository_domain_scores.csv
```

The plot values are exported separately to facilitate checking, reuse, and editorial revision. Figure 3 reporting-completeness counts are retained in the script as adjudicated manuscript values because several source fields contain narrative text rather than a single machine-readable response.

## Feature-extraction case study

### Input organisation

`reproducibility_casestudy_S5_LDH.py` expects a data directory containing ZIP archives, each with one or more WAV files. It recursively searches `DATA_DIR` for `*.zip` files and extracts only one WAV at a time to a temporary directory, limiting disk use.

Illustrative structure:

```text
case_study_data/
├── 300_P.zip
│   └── 300_P/
│       └── audio.wav
├── 301_P.zip
│   └── 301_P/
│       └── audio.wav
└── ...
```

Set the following configuration variables at the top of the script before running:

```python
DATA_DIR = Path("/path/to/case_study_data")
OUTPUT_DIR = Path("results")
MAX_FILES = None
```

The participant-ID extraction function uses DAIC-WOZ-style patterns. Adapt `extract_participant_id()` if your archive or WAV naming convention differs. Correct matching to outcomes depends on this step.

### Extraction settings

The script implements:

| Implementation | Configuration | Main outputs |
|---|---|---|
| openSMILE | `eGeMAPSv02` at the `Functionals` level | eGeMAPS-derived functionals prefixed `opensmile_` |
| Praat via Parselmouth | Pitch floor 50 Hz, pitch ceiling 500 Hz, 10 ms time step | Intensity, F0, HNR, local jitter, local shimmer, duration, sampling frequency prefixed `praat_` |
| librosa | Mono audio resampled to 16 kHz; FFT 1024; hop 256; 13 MFCCs; YIN F0 range 50–500 Hz | RMS, spectral centroid/bandwidth, zero-crossing rate, MFCC mean/SD, F0, duration, sampling frequency prefixed `librosa_` |

The default minimum recording duration is 0.5 seconds. Change parameter values only deliberately and report them with downstream results.

### Run extraction

```bash
python reproducibility_casestudy_S5_LDH.py
```

The script creates:

```text
results/
├── feature_values_checkpoint.csv
├── feature_values_long.csv
├── feature_summary_by_method.csv
├── feature_agreement.csv
├── file_processing_log.csv
└── extraction_errors.csv
```

`feature_values_long.csv` is the central recording-level table. It contains metadata (`participant_id`, `recording_id`, ZIP filename, and WAV path within the archive) plus feature columns with implementation-specific prefixes.

`feature_agreement.csv` compares nominally comparable measures when at least 20 finite pairs are available. It includes Pearson and Spearman associations, absolute and relative differences, standardised mean absolute error, regression slope/intercept, and method-specific summary values.

## Outcome reproducibility analysis

### Required inputs

`reproducibility_outcomes_S5_LDH-2.py` requires:

- `results/feature_values_long.csv` from the extraction script.
- A development outcome CSV.
- A test outcome CSV.

By default, the script expects AVEC 2017-compatible columns:

```text
Participant_ID
PHQ_Binary
```

It also uses `PHQ_Score` and `Gender` if those columns are present. The development and test outcome files are concatenated after retaining the required columns, and participant IDs are normalised before merging.

Set the paths and, if necessary, column names at the top of the script:

```python
RESULTS_DIR = Path("/path/to/results")
DEV_OUTCOME_FILE = Path("/path/to/dev_split_Depression_AVEC2017.csv")
TEST_OUTCOME_FILE = Path("/path/to/full_test_split.csv")
OUTCOME_PARTICIPANT_COL = "Participant_ID"
OUTCOME_COL = "PHQ_Binary"
```

### Run analysis

```bash
python reproducibility_outcomes_S5_LDH-2.py
```

The script writes to:

```text
results/reproducibility/
├── analysis_configuration.json
├── tables/
│   ├── S1_extraction_summary.csv
│   ├── S2_feature_agreement.csv
│   ├── S3_prediction_performance.csv
│   ├── S4_feature_selection.csv
│   ├── S5_feature_overlap.csv
│   └── participant_level_oof_predictions.csv
└── figures/
    ├── Fig_S1_feature_agreement.png
    ├── Fig_S2_BlandAltman_*.png
    ├── Fig_S3_prediction_performance.png
    └── Fig_S4_feature_selection_overlap.png
```

### Analytic design

The outcome analysis is designed to isolate the influence of feature-extraction implementation. For each available method-specific feature set, it applies the same participant sample, repeated stratified cross-validation splits, preprocessing, feature-selection procedure, classifier, and outcome definition.

The default model protocol is:

- Participant-level features formed by averaging recording-level acoustic features within participant.
- Repeated stratified 5-fold cross-validation with 10 repeats and `random_state = 42`.
- Median imputation, standard scaling, univariate `SelectKBest(f_classif)` feature selection, and logistic regression (`liblinear`, maximum 3,000 iterations).
- Selection of up to 20 features within each training fold.
- Repeated out-of-fold probabilities averaged for each participant.
- Participant-level nonparametric bootstrap 95% confidence intervals using 2,000 resamples.
- Feature-selection frequency across cross-validation splits and Jaccard overlap of features selected in at least 50% of splits.

The script produces ROC AUC, accuracy, balanced accuracy, sensitivity, specificity, precision, and F1 score. Treat these analyses as an illustrative reproducibility comparison rather than a clinical validation study.

## Reproduction order

For the full project, run the scripts in this order:

1. Prepare the review publication CSV, repository workbook, and any adjudication override files.
2. Run `analysis_consolidation_LDH-4.py` to create the systematic-review tables, figures, workbook, provenance manifest, and QA outputs.
3. Optionally run `figures_LDH-3.py` to generate the alternative Figure 2 and Figure 3 designs from the review data.
4. Obtain authorised case-study ZIP archives containing WAV files, set `DATA_DIR`, and run `reproducibility_casestudy_S5_LDH.py`.
5. Set `RESULTS_DIR` and authorised outcome-file paths, then run `reproducibility_outcomes_S5_LDH-2.py`.
6. Review `extraction_errors.csv`, `file_processing_log.csv`, ID-matching messages, QA files, configuration JSON, and source-data CSVs before interpreting or submitting outputs.

## Interpretation notes

- Software labels and feature descriptions in published articles are heterogeneous. The systematic-review scripts use transparent, rule-based text harmonisation to organise these fields; they do not replace manual adjudication.
- A publication can contribute to multiple diagnosis, task, feature-family, software, language, and validation categories. Category totals therefore need not sum to the number of included publications.
- “Open code” and “open data” are classified separately from “available on request.” The latter is not treated as publicly available.
- Similar feature names across libraries may differ because of preprocessing, voiced-frame handling, windowing, parameter defaults, algorithms, summary functionals, units, and numerical implementation.
- Correlation quantifies covariation, not interchangeability. The agreement tables and Bland–Altman plots should be considered together with absolute/relative differences and downstream prediction results.
- The output scripts are intended for reproducible analysis, but reruns may vary if package versions, operating systems, source datasets, or manual override files differ.

## Citation

If you use this code, please cite the associated systematic review:

> *Reproducibility of acoustic speech feature extraction in psychiatry: a systematic review of methods, software and open-source repositories.*

Please also cite the original software and dataset resources used in any analysis, including openSMILE, Praat/Parselmouth, librosa, scikit-learn, and the authorised source of the speech/outcome data.

## Licence and contact

Add the project licence before public release (for example, `LICENSE`). Replace this section with the corresponding author/contact details and the final manuscript DOI or preprint link when available.
