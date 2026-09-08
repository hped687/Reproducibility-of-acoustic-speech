"""
====================================================================
SPEECH FEATURE EXTRACTION REPRODUCIBILITY ANALYSIS
====================================================================

Purpose
-------
Extract comparable acoustic features from every WAV file contained
inside participant-specific ZIP archives.

Extraction implementations
---------------------------
1. openSMILE - eGeMAPSv02
2. Praat     - via Parselmouth
3. librosa


Outputs
-------
results/
    feature_values_long.csv
    feature_summary_by_method.csv
    feature_agreement.csv
    file_processing_log.csv
    extraction_errors.csv

The long-format file contains one row per:
    participant x recording x method

The agreement file compares nominally equivalent features between
methods.

====================================================================
"""

from __future__ import annotations

import os
import re
import gc
import json
import math
import shutil
import zipfile
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Optional imports
# ------------------------------------------------------------------

try:
    import opensmile
    HAVE_OPENSMILE = True
except ImportError:
    HAVE_OPENSMILE = False

try:
    import parselmouth
    from parselmouth.praat import call
    HAVE_PRAAT = True
except ImportError:
    HAVE_PRAAT = False

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False

try:
    from scipy.stats import pearsonr, spearmanr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ==================================================================
# USER SETTINGS
# ==================================================================

DATA_DIR = Path("") #insert data directory path
OUTPUT_DIR = Path("results")

# Temporary extraction directory.
# Only ONE WAV is extracted at a time.
TEMP_DIR = OUTPUT_DIR / "_tmp"

# Minimum duration to process.
MIN_DURATION_SEC = 0.5

# Maximum number of WAV files.
# Set to None for the entire dataset.
MAX_FILES = None

# ------------------------------------------------------------------
# librosa settings
# ------------------------------------------------------------------

LIBROSA_SR = 16000

N_FFT = 1024
HOP_LENGTH = 256

N_MFCC = 13

# F0 range.
F0_MIN = 50
F0_MAX = 500

# ------------------------------------------------------------------
# Praat settings
# ------------------------------------------------------------------

PRAAT_TIME_STEP = 0.01
PRAAT_F0_MIN = 50
PRAAT_F0_MAX = 500

# ------------------------------------------------------------------
# Agreement analysis
# ------------------------------------------------------------------

MIN_PAIRS_FOR_COMPARISON = 20

# Features considered nominally equivalent.
#
# IMPORTANT:
# These are NOT assumed to be mathematically identical.
# They are constructs for which different implementations exist.

COMPARABLE_FEATURES = {
    "f0": [
        "opensmile_F0final_sma3nz_amean",
        "praat_f0_mean",
        "librosa_f0_mean",
    ],

    "jitter": [
        "opensmile_jitterLocal_sma3nz_amean",
        "praat_jitter_local",
    ],

    "shimmer": [
        "opensmile_shimmerLocaldB_sma3nz_amean",
        "praat_shimmer_local",
    ],

    "hnr": [
        "opensmile_HNRdBACF_sma3nz_amean",
        "praat_hnr_mean",
    ],

    "loudness": [
        "opensmile_loudness_sma3_amean",
        "praat_intensity_mean",
        "librosa_rms_mean",
    ],

    "mfcc_1": [
        "opensmile_mfcc1_sma3_amean",
        "librosa_mfcc_1_mean",
    ],

    "mfcc_2": [
        "opensmile_mfcc2_sma3_amean",
        "librosa_mfcc_2_mean",
    ],

    "mfcc_3": [
        "opensmile_mfcc3_sma3_amean",
        "librosa_mfcc_3_mean",
    ],

    "mfcc_4": [
        "opensmile_mfcc4_sma3_amean",
        "librosa_mfcc_4_mean",
    ],

    "mfcc_5": [
        "opensmile_mfcc5_sma3_amean",
        "librosa_mfcc_5_mean",
    ],

    "mfcc_6": [
        "opensmile_mfcc6_sma3_amean",
        "librosa_mfcc_6_mean",
    ],

    "mfcc_7": [
        "opensmile_mfcc7_sma3_amean",
        "librosa_mfcc_7_mean",
    ],

    "mfcc_8": [
        "opensmile_mfcc8_sma3_amean",
        "librosa_mfcc_8_mean",
    ],

    "mfcc_9": [
        "opensmile_mfcc9_sma3_amean",
        "librosa_mfcc_9_mean",
    ],

    "mfcc_10": [
        "opensmile_mfcc10_sma3_amean",
        "librosa_mfcc_10_mean",
    ],

    "mfcc_11": [
        "opensmile_mfcc11_sma3_amean",
        "librosa_mfcc_11_mean",
    ],

    "mfcc_12": [
        "opensmile_mfcc12_sma3_amean",
        "librosa_mfcc_12_mean",
    ],

    "mfcc_13": [
        "opensmile_mfcc13_sma3_amean",
        "librosa_mfcc_13_mean",
    ],
}


# ==================================================================
# SETUP
# ==================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SPEECH FEATURE EXTRACTION REPRODUCIBILITY ANALYSIS")
print("=" * 80)

print("\nAvailable extraction engines:")

print(f"  openSMILE : {HAVE_OPENSMILE}")
print(f"  Praat     : {HAVE_PRAAT}")
print(f"  librosa   : {HAVE_LIBROSA}")

if not any([HAVE_OPENSMILE, HAVE_PRAAT, HAVE_LIBROSA]):
    raise RuntimeError(
        "None of openSMILE, Praat/Parselmouth or librosa are installed."
    )


# ==================================================================
# INITIALISE openSMILE
# ==================================================================

smile = None

if HAVE_OPENSMILE:

    print("\nInitialising openSMILE eGeMAPSv02...")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    print("  openSMILE ready.")


# ==================================================================
# UTILITY FUNCTIONS
# ==================================================================

def clean_number(x):
    """Convert numerical values to finite float or NaN."""

    try:
        x = float(x)

        if np.isfinite(x):
            return x

    except Exception:
        pass

    return np.nan


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.median(x))


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.std(x, ddof=1))


def extract_participant_id(zip_path, wav_name):
    """
    Try to recover participant ID from the ZIP filename or WAV path.

    Adapt this function if your DAIC-WOZ naming convention differs.
    """

    text = f"{zip_path.stem}_{wav_name}"

    # Common DAIC-WOZ-style identifiers.
    patterns = [
        r"(?:participant|Participant)[_\-]?(\d+)",
        r"(?:session|Session)[_\-]?(\d+)",
        r"\b(\d{3,5})\b",
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return zip_path.stem


def get_wav_files_from_zip(zip_path):

    with zipfile.ZipFile(zip_path, "r") as z:

        names = [
            name for name in z.namelist()
            if name.lower().endswith(".wav")
            and not name.endswith("/")
        ]

    return names


def extract_single_wav(zip_path, wav_name, destination):

    """
    Extract exactly one WAV file.

    This is intentional: we do not unzip the entire participant
    archive, keeping disk and memory usage low.
    """

    with zipfile.ZipFile(zip_path, "r") as z:

        with z.open(wav_name) as source:

            with open(destination, "wb") as target:

                shutil.copyfileobj(source, target, length=1024 * 1024)


# ==================================================================
# openSMILE
# ==================================================================

def extract_opensmile(wav_path):

    if not HAVE_OPENSMILE:

        return {}

    try:

        df = smile.process_file(str(wav_path))

        if df.empty:
            return {}

        row = df.iloc[0]

        result = {}

        for feature, value in row.items():

            result[f"opensmile_{feature}"] = clean_number(value)

        return result

    except Exception as e:

        print(f"    openSMILE ERROR: {e}")

        return {}


# ==================================================================
# PRAAT
# ==================================================================

def extract_praat(wav_path):

    if not HAVE_PRAAT:

        return {}

    result = {}

    try:

        sound = parselmouth.Sound(str(wav_path))

        duration = sound.duration

        # ----------------------------------------------------------
        # Intensity
        # ----------------------------------------------------------

        try:

            intensity = sound.to_intensity(
                minimum_pitch=PRAAT_F0_MIN,
                time_step=PRAAT_TIME_STEP,
            )

            intensity_values = intensity.values[0]

            result["praat_intensity_mean"] = safe_mean(
                intensity_values
            )

            result["praat_intensity_sd"] = safe_std(
                intensity_values
            )

        except Exception:

            result["praat_intensity_mean"] = np.nan
            result["praat_intensity_sd"] = np.nan

        # ----------------------------------------------------------
        # Pitch
        # ----------------------------------------------------------

        try:

            pitch = sound.to_pitch(
                time_step=PRAAT_TIME_STEP,
                pitch_floor=PRAAT_F0_MIN,
                pitch_ceiling=PRAAT_F0_MAX,
            )

            values = pitch.selected_array["frequency"]

            values = values[
                np.isfinite(values) &
                (values > 0)
            ]

            result["praat_f0_mean"] = safe_mean(values)
            result["praat_f0_median"] = safe_median(values)
            result["praat_f0_sd"] = safe_std(values)

        except Exception:

            result["praat_f0_mean"] = np.nan
            result["praat_f0_median"] = np.nan
            result["praat_f0_sd"] = np.nan

        # ----------------------------------------------------------
        # Harmonics-to-noise ratio
        # ----------------------------------------------------------

        try:

            harmonicity = sound.to_harmonicity_cc(
                time_step=PRAAT_TIME_STEP,
                minimum_pitch=PRAAT_F0_MIN,
            )

            values = harmonicity.values[0]

            values = values[
                np.isfinite(values) &
                (values > -100)
            ]

            result["praat_hnr_mean"] = safe_mean(values)

        except Exception:

            result["praat_hnr_mean"] = np.nan

        # ----------------------------------------------------------
        # Jitter and shimmer
        # ----------------------------------------------------------

        try:

            point_process = call(
                sound,
                "To PointProcess (periodic, cc)",
                PRAAT_F0_MIN,
                PRAAT_F0_MAX,
            )

            result["praat_jitter_local"] = clean_number(
                call(
                    point_process,
                    "Get jitter (local)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                )
            )

            result["praat_shimmer_local"] = clean_number(
                call(
                    [sound, point_process],
                    "Get shimmer (local)",
                    0,
                    0,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )

        except Exception:

            result["praat_jitter_local"] = np.nan
            result["praat_shimmer_local"] = np.nan

        # ----------------------------------------------------------
        # Basic audio information
        # ----------------------------------------------------------

        result["praat_duration_sec"] = duration

        result["praat_sampling_frequency"] = sound.sampling_frequency

        # Explicit cleanup.
        del sound

        return result

    except Exception as e:

        print(f"    Praat ERROR: {e}")

        return result


# ==================================================================
# LIBROSA
# ==================================================================

def extract_librosa(wav_path):

    if not HAVE_LIBROSA:

        return {}

    result = {}

    try:

        y, sr = librosa.load(
            str(wav_path),
            sr=LIBROSA_SR,
            mono=True,
        )

        if len(y) == 0:

            return result

        duration = len(y) / sr

        result["librosa_duration_sec"] = duration

        result["librosa_sampling_frequency"] = sr

        # ----------------------------------------------------------
        # RMS energy
        # ----------------------------------------------------------

        rms = librosa.feature.rms(
            y=y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]

        result["librosa_rms_mean"] = safe_mean(rms)
        result["librosa_rms_sd"] = safe_std(rms)

        # ----------------------------------------------------------
        # Spectral centroid
        # ----------------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]

        result["librosa_spectral_centroid_mean"] = safe_mean(
            centroid
        )

        result["librosa_spectral_centroid_sd"] = safe_std(
            centroid
        )

        # ----------------------------------------------------------
        # Spectral bandwidth
        # ----------------------------------------------------------

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]

        result["librosa_spectral_bandwidth_mean"] = safe_mean(
            bandwidth
        )

        # ----------------------------------------------------------
        # Zero-crossing rate
        # ----------------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]

        result["librosa_zcr_mean"] = safe_mean(zcr)

        # ----------------------------------------------------------
        # MFCC
        # ----------------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        for i in range(N_MFCC):

            result[
                f"librosa_mfcc_{i + 1}_mean"
            ] = safe_mean(mfcc[i])

            result[
                f"librosa_mfcc_{i + 1}_sd"
            ] = safe_std(mfcc[i])

        # ----------------------------------------------------------
        # F0
        # ----------------------------------------------------------

        try:

            f0 = librosa.yin(
                y,
                fmin=F0_MIN,
                fmax=F0_MAX,
                sr=sr,
                frame_length=N_FFT,
                hop_length=HOP_LENGTH,
            )

            f0 = f0[
                np.isfinite(f0) &
                (f0 > 0)
            ]

            result["librosa_f0_mean"] = safe_mean(f0)
            result["librosa_f0_median"] = safe_median(f0)
            result["librosa_f0_sd"] = safe_std(f0)

        except Exception:

            result["librosa_f0_mean"] = np.nan
            result["librosa_f0_median"] = np.nan
            result["librosa_f0_sd"] = np.nan

        # Explicitly release audio arrays.
        del y
        del mfcc
        del rms
        del centroid
        del bandwidth
        del zcr

        gc.collect()

        return result

    except Exception as e:

        print(f"    librosa ERROR: {e}")

        return result


# ==================================================================
# SINGLE WAV
# ==================================================================

def process_wav(zip_path, wav_name, temp_wav):

    participant_id = extract_participant_id(
        zip_path,
        wav_name,
    )

    recording_id = Path(wav_name).stem

    result = {
        "participant_id": participant_id,
        "recording_id": recording_id,
        "zip_file": zip_path.name,
        "wav_path_in_zip": wav_name,
    }

    # --------------------------------------------------------------
    # Process using each implementation
    # --------------------------------------------------------------

    if HAVE_OPENSMILE:

        result.update(
            extract_opensmile(temp_wav)
        )

        gc.collect()

    if HAVE_PRAAT:

        result.update(
            extract_praat(temp_wav)
        )

        gc.collect()

    if HAVE_LIBROSA:

        result.update(
            extract_librosa(temp_wav)
        )

        gc.collect()

    return result


# ==================================================================
# FIND ZIP FILES
# ==================================================================

zip_files = sorted(
    DATA_DIR.rglob("*.zip")
)

print("\nZIP archives found:", len(zip_files))

if len(zip_files) == 0:

    raise FileNotFoundError(
        f"No ZIP files found under {DATA_DIR.resolve()}"
    )


# ==================================================================
# PROCESS DATASET
# ==================================================================

all_results = []
processing_log = []
errors = []

file_counter = 0

for zip_index, zip_path in enumerate(zip_files, start=1):

    print("\n" + "=" * 80)
    print(
        f"ZIP {zip_index}/{len(zip_files)}: "
        f"{zip_path.name}"
    )
    print("=" * 80)

    try:

        wav_files = get_wav_files_from_zip(zip_path)

    except Exception as e:

        print("Could not read ZIP:", e)

        errors.append({
            "zip_file": zip_path.name,
            "error_type": "zip_read",
            "error": str(e),
        })

        continue

    print("WAV files:", len(wav_files))

    for wav_index, wav_name in enumerate(wav_files, start=1):

        if (
            MAX_FILES is not None
            and file_counter >= MAX_FILES
        ):
            break

        file_counter += 1

        print(
            f"\n[{file_counter}] "
            f"{wav_index}/{len(wav_files)} "
            f"{wav_name}"
        )

        temp_wav = (
            TEMP_DIR /
            f"current_{os.getpid()}.wav"
        )

        try:

            # ------------------------------------------------------
            # Extract ONE WAV
            # ------------------------------------------------------

            extract_single_wav(
                zip_path,
                wav_name,
                temp_wav,
            )

            # ------------------------------------------------------
            # Process ONE WAV
            # ------------------------------------------------------

            result = process_wav(
                zip_path,
                wav_name,
                temp_wav,
            )

            all_results.append(result)

            processing_log.append({
                "zip_file": zip_path.name,
                "wav_file": wav_name,
                "status": "success",
            })

            # ------------------------------------------------------
            # Delete temporary WAV immediately
            # ------------------------------------------------------

            if temp_wav.exists():
                temp_wav.unlink()

            gc.collect()

        except Exception as e:

            print("ERROR:", e)

            errors.append({
                "zip_file": zip_path.name,
                "wav_file": wav_name,
                "error_type": "processing",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

            processing_log.append({
                "zip_file": zip_path.name,
                "wav_file": wav_name,
                "status": "error",
            })

            if temp_wav.exists():
                try:
                    temp_wav.unlink()
                except Exception:
                    pass

            gc.collect()

    # --------------------------------------------------------------
    # Save intermediate checkpoint after every ZIP
    # --------------------------------------------------------------

    if all_results:

        checkpoint = pd.DataFrame(all_results)

        checkpoint.to_csv(
            OUTPUT_DIR /
            "feature_values_checkpoint.csv",
            index=False,
        )

        del checkpoint

        gc.collect()


# ==================================================================
# FINAL FEATURE TABLE
# ==================================================================

print("\n" + "=" * 80)
print("CREATING FINAL FEATURE TABLE")
print("=" * 80)

features = pd.DataFrame(all_results)

features.to_csv(
    OUTPUT_DIR /
    "feature_values_long.csv",
    index=False,
)

print(
    "Feature table:",
    features.shape,
)


# ==================================================================
# FEATURE SUMMARY
# ==================================================================

metadata_cols = [
    "participant_id",
    "recording_id",
    "zip_file",
    "wav_path_in_zip",
]

feature_columns = [
    c for c in features.columns
    if c not in metadata_cols
]

summary_rows = []

for feature in feature_columns:

    values = pd.to_numeric(
        features[feature],
        errors="coerce",
    )

    summary_rows.append({
        "feature": feature,
        "n": int(values.notna().sum()),
        "mean": values.mean(),
        "sd": values.std(),
        "median": values.median(),
        "min": values.min(),
        "max": values.max(),
        "missing_pct": values.isna().mean() * 100,
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR /
    "feature_summary_by_method.csv",
    index=False,
)


# ==================================================================
# METHOD AGREEMENT
# ==================================================================

print("\n" + "=" * 80)
print("CALCULATING CROSS-IMPLEMENTATION AGREEMENT")
print("=" * 80)


def calculate_agreement(
    df,
    feature_a,
    feature_b,
    construct,
    method_a,
    method_b,
):

    if feature_a not in df.columns:
        return None

    if feature_b not in df.columns:
        return None

    x = pd.to_numeric(
        df[feature_a],
        errors="coerce",
    )

    y = pd.to_numeric(
        df[feature_b],
        errors="coerce",
    )

    valid = (
        np.isfinite(x) &
        np.isfinite(y)
    )

    x = x[valid].values
    y = y[valid].values

    if len(x) < MIN_PAIRS_FOR_COMPARISON:

        return None

    # --------------------------------------------------------------
    # Pearson
    # --------------------------------------------------------------

    if HAVE_SCIPY:

        try:
            pearson_r, pearson_p = pearsonr(x, y)
        except Exception:
            pearson_r = np.nan
            pearson_p = np.nan

        try:
            spearman_r, spearman_p = spearmanr(x, y)
        except Exception:
            spearman_r = np.nan
            spearman_p = np.nan

    else:

        pearson_r = np.corrcoef(x, y)[0, 1]
        pearson_p = np.nan

        spearman_r = np.nan
        spearman_p = np.nan

    # --------------------------------------------------------------
    # Difference measures
    # --------------------------------------------------------------

    absolute_difference = np.abs(x - y)

    mean_abs_difference = np.mean(
        absolute_difference
    )

    median_abs_difference = np.median(
        absolute_difference
    )

    # --------------------------------------------------------------
    # Relative difference
    #
    # Avoid division by zero.
    # --------------------------------------------------------------

    denominator = np.maximum(
        np.abs(x),
        np.finfo(float).eps,
    )

    relative_difference = (
        np.abs(x - y) / denominator
    )

    # --------------------------------------------------------------
    # Standardised difference
    #
    # Useful because feature scales differ dramatically.
    # --------------------------------------------------------------

    pooled_sd = np.sqrt(
        (
            np.var(x, ddof=1)
            +
            np.var(y, ddof=1)
        ) / 2
    )

    if pooled_sd > 0:

        standardised_mae = (
            mean_abs_difference /
            pooled_sd
        )

    else:

        standardised_mae = np.nan

    # --------------------------------------------------------------
    # Regression slope
    # --------------------------------------------------------------

    try:

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

    except Exception:

        slope = np.nan
        intercept = np.nan

    return {
        "construct": construct,
        "method_a": method_a,
        "method_b": method_b,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "n": len(x),

        "pearson_r": pearson_r,
        "pearson_p": pearson_p,

        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,

        "mean_absolute_difference":
            mean_abs_difference,

        "median_absolute_difference":
            median_abs_difference,

        "median_relative_difference":
            np.median(relative_difference),

        "standardised_mae":
            standardised_mae,

        "regression_slope":
            slope,

        "regression_intercept":
            intercept,

        "mean_a": np.mean(x),
        "mean_b": np.mean(y),

        "sd_a": np.std(x, ddof=1),
        "sd_b": np.std(y, ddof=1),
    }


agreement_rows = []


for construct, feature_names in COMPARABLE_FEATURES.items():

    # Remove features not available in the current run.
    available = [
        f for f in feature_names
        if f in features.columns
    ]

    if len(available) < 2:
        continue

    # Compare every available implementation.
    for i in range(len(available)):

        for j in range(i + 1, len(available)):

            feature_a = available[i]
            feature_b = available[j]

            method_a = feature_a.split("_")[0]
            method_b = feature_b.split("_")[0]

            result = calculate_agreement(
                features,
                feature_a,
                feature_b,
                construct,
                method_a,
                method_b,
            )

            if result is not None:

                agreement_rows.append(result)


agreement = pd.DataFrame(
    agreement_rows
)

agreement.to_csv(
    OUTPUT_DIR /
    "feature_agreement.csv",
    index=False,
)

print(
    "Agreement table:",
    agreement.shape,
)


# ==================================================================
# PROCESSING LOG
# ==================================================================

pd.DataFrame(
    processing_log
).to_csv(
    OUTPUT_DIR /
    "file_processing_log.csv",
    index=False,
)


pd.DataFrame(
    errors
).to_csv(
    OUTPUT_DIR /
    "extraction_errors.csv",
    index=False,
)


# ==================================================================
# FINAL REPORT
# ==================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nZIP archives processed: {len(zip_files)}"
)

print(
    f"WAV files attempted: {file_counter}"
)

print(
    f"Successful WAV files: {len(all_results)}"
)

print(
    f"Errors: {len(errors)}"
)

print(
    f"\nResults written to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nMain outputs:")

print(
    "  feature_values_long.csv"
)

print(
    "  feature_summary_by_method.csv"
)

print(
    "  feature_agreement.csv"
)

print(
    "  file_processing_log.csv"
)

print(
    "  extraction_errors.csv"
)

print(
    "\nTemporary files are removed after each WAV."
)

print("=" * 80)
