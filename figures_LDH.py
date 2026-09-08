#!/usr/bin/env python3
"""Create the Figures from the review data files.

Outputs
-------
- Figure_2_alternative_tile_matrices.png/.pdf/.svg
- Figure_3_alternative_reproducibility.png/.pdf/.svg
- Figure_2A_feature_counts.csv
- Figure_2A_feature_percentages.csv
- Figure_2B_software_counts.csv
- Figure_2B_software_percentages.csv
- Figure_3A_reporting_completeness.csv
- Figure_3B_repository_domain_scores.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Default local folders
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(
    r""
)
DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

PUBLICATION_FILENAME = "28Jul26_FINAL_v2.csv"
GITHUB_FILENAME = "28Jul26_FINAL_GITHUB.xlsx"


# -----------------------------------------------------------------------------
# Harmonisation rules
# A publication may contribute to more than one diagnosis, feature family or
# software category.
# -----------------------------------------------------------------------------
DIAGNOSIS_PATTERNS: Dict[str, str] = {
    "Depression": r"\bmdd\b|depress",
    "Psychosis / schizophrenia": r"schiz|psychosis|schizoaffective|schizotyp",
    "Autism": r"autis|\basd\b|broad autism phenotype",
    "Anxiety": r"anxiety|panic|agoraphob|phobia",
    "Bipolar disorder": r"bipolar",
    "PTSD / trauma": r"\bptsd\b|post.?trauma|trauma",
    "Suicidality": r"suicid",
    "ADHD": r"\badhd\b|hyperactivity|inattention",
    "Eating disorders": r"anorex|bulim|eating disorder",
    "OCD": r"\bocd\b|obsessive",
    "Personality disorders": r"personality|borderline",
}

FEATURE_PATTERNS: Dict[str, str] = {
    "Prosodic": r"prosod|pitch|fundamental frequency|\bf0\b|intonation|rhythm",
    "Energy / intensity": r"\benergy\b|intensity|loudness|\brms\b",
    "Spectral": (
        r"spectral|spectrogram|\bstft\b|\bfft\b|filter.?bank|\bfbank\b|"
        r"log.?mel|\blpc\b|wavelet|\bwpt\b|\bgtcc\b|\bplp\b"
    ),
    "Cepstral": r"cepstr|mfcc|mel.frequency cepstr",
    "Temporal / pausing": (
        r"temporal|pause|speech rate|articulation rate|latency|turn.taking|"
        r"duration|rhythm|voiced.*unvoiced"
    ),
    "Voice quality / perturbation": (
        r"jitter|shimmer|harmonic.*noise|\bhnr\b|voice quality|perturb|"
        r"\bcpp\b|glottal"
    ),
    "Formant / articulatory": (
        r"formant|articul|vowel space|\bvsa\b|\bf1\b|\bf2\b|\bf3\b|\bf4\b"
    ),
    "Learned representation": (
        r"wav2vec|hubert|wavlm|whisper|embedding|deep feature|"
        r"self.supervised|learned representation|\bssl\b|end-to-end|"
        r"spectrogram image features learned|convolutional autoencoder"
    ),
}

SOFTWARE_PATTERNS: Dict[str, str] = {
    "openSMILE": r"opensmile",
    "Praat / Parselmouth": r"praat|parselmouth",
    "Custom / in-house / proprietary": (
        r"custom|in.house|proprietary|self.developed|bespoke"
    ),
    "librosa": r"librosa",
    "MATLAB": r"matlab",
    "COVAREP": r"covarep",
    "Kaldi": r"kaldi",
    "PyAudioAnalysis": r"pyaudioanalysis|py audio analysis",
    "DisVoice": r"disvoice",
    "Self-supervised models": r"wav2vec|hubert|wavlm|whisper",
}

REPOSITORY_DOMAIN_LABELS: Dict[str, str] = {
    "q_scope_clarity": "Scope clarity",
    "q_installation": "Installation instructions",
    "q_documentation": "Documentation",
    "q_feature_transparency": "Feature transparency",
    "q_parameter_transparency": "Parameter transparency",
    "q_reproducible_example": "Reproducible example",
    "q_versioning": "Versioning / releases",
    "q_testing": "Automated testing / CI",
    "q_maintenance": "Maintenance",
    "q_license": "Licence",
    "q_publication_linkage": "Publication linkage",
    "q_clinical_relevance": "Clinical relevance",
}

# These are the adjudicated counts used in the current manuscript. They are retained because several extraction fields contain narrative
# text rather than a single machine-readable Yes/No variable.
ADJUDICATED_REPORTING_COUNTS = {
    "Recording environment": (277, "Input"),
    "Device / microphone": (199, "Input"),
    "Sampling rate": (169, "Input"),
    "Segmentation / VAD": (267, "Preprocessing"),
    "Noise reduction / filtering": (121, "Preprocessing"),
    "Software version / commit": (143, "Extraction"),
    "Frame length": (157, "Extraction"),
    "Window / hop setting": (123, "Extraction"),
    "Sampling + frame + window complete": (71, "Extraction"),
    "Public code": (40, "Implementation"),
    "Version + complete configuration": (29, "Implementation"),
    "External / cross-corpus validation": (29, "Validation"),
    "Participant-independent split stated": (27, "Validation"),
    "Test-retest / reliability assessment": (23, "Validation"),
}



# Short display labels used only in Figure 3. The source-data CSVs retain the
# full manuscript terminology.
FIGURE3A_SHORT_LABELS: Dict[str, str] = {
    "Recording environment": "Environment",
    "Device / microphone": "Device / mic",
    "Sampling rate": "Sampling rate",
    "Segmentation / VAD": "Segmentation / VAD",
    "Noise reduction / filtering": "Denoising / filtering",
    "Software version / commit": "Software version",
    "Frame length": "Frame length",
    "Window / hop setting": "Window / hop",
    "Sampling + frame + window complete": "Complete signal settings",
    "Public code": "Public code",
    "Version + complete configuration": "Version + full configuration",
    "External / cross-corpus validation": "External validation",
    "Participant-independent split stated": "Participant-level split",
    "Test-retest / reliability assessment": "Test-retest reliability",
}

FIGURE3B_SHORT_LABELS: Dict[str, str] = {
    "Scope clarity": "Scope",
    "Installation instructions": "Installation",
    "Documentation": "Documentation",
    "Feature transparency": "Feature details",
    "Parameter transparency": "Parameters",
    "Reproducible example": "Worked example",
    "Versioning / releases": "Versioning",
    "Automated testing / CI": "Testing / CI",
    "Maintenance": "Maintenance",
    "Licence": "Licence",
    "Publication linkage": "Publication link",
    "Clinical relevance": "Clinical relevance",
}

PIPELINE_ORDER = [
    "Input",
    "Preprocessing",
    "Extraction",
    "Implementation",
    "Validation",
]


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------
def locate_input_file(data_dir: Path, preferred_name: str, pattern: str) -> Path:
    """Use the expected filename, or fall back to one unambiguous match."""
    preferred = data_dir / preferred_name
    if preferred.exists():
        return preferred

    matches = sorted(data_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{preferred_name}' in:\n{data_dir}\n"
            f"Expected a file matching: {pattern}"
        )
    raise FileNotFoundError(
        f"Could not choose a single input for '{preferred_name}'. "
        f"Found: {[p.name for p in matches]}"
    )


def validate_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"The following required columns are missing from {source.name}: {missing}"
        )


def load_inputs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    publication_file = locate_input_file(
        data_dir,
        PUBLICATION_FILENAME,
        "*FINAL*.csv",
    )
    github_file = locate_input_file(
        data_dir,
        GITHUB_FILENAME,
        "*FINAL*GITHUB*.xlsx",
    )

    publications = pd.read_csv(publication_file, low_memory=False)
    quality = pd.read_excel(github_file, sheet_name="Quality_Appraisal")

    validate_columns(
        publications,
        {"Diagnostic Group(s)", "Feature(s)", "Software / tool / package"},
        publication_file,
    )
    validate_columns(
        quality,
        set(REPOSITORY_DOMAIN_LABELS)
        | {"appraisal_complete", "quality_total_scored", "repository_class"},
        github_file,
    )

    return publications, quality, publication_file, github_file


# -----------------------------------------------------------------------------
# Source-data construction
# -----------------------------------------------------------------------------
def lower_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower()


def boolean_matrix(series: pd.Series, patterns: Dict[str, str]) -> pd.DataFrame:
    text = lower_text(series)
    return pd.DataFrame(
        {
            label: text.str.contains(pattern, regex=True, na=False)
            for label, pattern in patterns.items()
        },
        index=series.index,
    )


def build_figure2_data(
    publications: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    diagnoses = boolean_matrix(publications["Diagnostic Group(s)"], DIAGNOSIS_PATTERNS)
    features = boolean_matrix(publications["Feature(s)"], FEATURE_PATTERNS)
    software = boolean_matrix(
        publications["Software / tool / package"], SOFTWARE_PATTERNS
    )

    diagnosis_n = diagnoses.sum(axis=0).astype(int).to_dict()
    diagnosis_order = sorted(
        diagnoses.columns,
        key=lambda label: (-diagnosis_n[label], label),
    )

    feature_counts = pd.DataFrame(
        0,
        index=diagnosis_order,
        columns=list(FEATURE_PATTERNS),
        dtype=int,
    )
    software_counts = pd.DataFrame(
        0,
        index=diagnosis_order,
        columns=list(SOFTWARE_PATTERNS),
        dtype=int,
    )

    for diagnosis in diagnosis_order:
        for feature in feature_counts.columns:
            feature_counts.loc[diagnosis, feature] = int(
                (diagnoses[diagnosis] & features[feature]).sum()
            )
        for tool in software_counts.columns:
            software_counts.loc[diagnosis, tool] = int(
                (diagnoses[diagnosis] & software[tool]).sum()
            )

    denominators = pd.Series(diagnosis_n).loc[diagnosis_order]
    feature_percentages = feature_counts.div(denominators, axis=0)
    software_percentages = software_counts.div(denominators, axis=0)

    return (
        feature_counts,
        feature_percentages,
        software_counts,
        software_percentages,
        diagnosis_n,
    )


def build_reporting_data(n_publications: int) -> pd.DataFrame:
    rows = [
        {
            "Item": item,
            "n": count,
            "Percent": count / n_publications,
            "Domain": domain,
        }
        for item, (count, domain) in ADJUDICATED_REPORTING_COUNTS.items()
    ]
    reporting = pd.DataFrame(rows)
    reporting["Domain"] = pd.Categorical(
        reporting["Domain"], categories=PIPELINE_ORDER, ordered=True
    )
    return reporting.sort_values(
        ["Domain", "Percent"], ascending=[True, False]
    ).reset_index(drop=True)


def parse_complete(series: pd.Series) -> pd.Series:
    """Handle Boolean and text versions of appraisal_complete."""
    return series.eq(True) | lower_text(series).isin({"true", "yes", "1"})


def build_repository_quality_data(quality: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    complete = quality.loc[parse_complete(quality["appraisal_complete"])].copy()
    if complete.empty:
        raise ValueError("No complete repository appraisals were found.")

    rows = []
    for column, label in REPOSITORY_DOMAIN_LABELS.items():
        values = pd.to_numeric(complete[column], errors="coerce")
        rows.append(
            {
                "Domain": label,
                "Mean score": values.mean(),
                "Maximum score": 2,
                "n scored": int(values.notna().sum()),
                "Score 0": int(values.eq(0).sum()),
                "Score 1": int(values.eq(1).sum()),
                "Score 2": int(values.eq(2).sum()),
            }
        )

    return pd.DataFrame(rows), len(complete)


# -----------------------------------------------------------------------------
# Figure 2: labelled tile matrices
# -----------------------------------------------------------------------------
def tile_matrix(
    ax: plt.Axes,
    counts_df: pd.DataFrame,
    percentages_df: pd.DataFrame,
    diagnosis_n: Dict[str, int],
    panel_title: str,
    low_count_threshold: int = 10,
):
  
    rows = counts_df.index.tolist()
    columns = counts_df.columns.tolist()
    n_rows = len(rows)
    n_columns = len(columns)

    colour_map = plt.get_cmap("Blues")
    normaliser = Normalize(vmin=0, vmax=1)

    for row_number, row_label in enumerate(rows):
        for column_number, column_label in enumerate(columns):
            count = int(counts_df.loc[row_label, column_label])
            proportion = float(percentages_df.loc[row_label, column_label])

            if count == 0:
                face_colour = "white"
                text_colour = "black"
            else:
                face_colour = colour_map(normaliser(proportion))
                # White labels are easier to read on darker cells.
                text_colour = "white" if proportion >= 0.55 else "#1F1F1F"

            ax.add_patch(
                Rectangle(
                    (column_number - 0.5, row_number - 0.5),
                    1,
                    1,
                    facecolor=face_colour,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=1,
                )
            )

            # Leave zero cells blank. All non-zero cells have a directly readable
            # count and percentage, so readers do not need to estimate bubble area.
            if count > 0:
                count_label = f"{count}*" if count < low_count_threshold else str(count)
                ax.text(
                    column_number,
                    row_number - 0.10,
                    count_label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_colour,
                    zorder=2,
                )
                ax.text(
                    column_number,
                    row_number + 0.19,
                    f"{proportion * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=text_colour,
                    zorder=2,
                )

    y_labels = [f"{label} (n={diagnosis_n[label]})" for label in rows]
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels(columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlim(-0.5, n_columns - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("auto")
    ax.set_title(panel_title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


def create_figure2(
    feature_counts: pd.DataFrame,
    feature_percentages: pd.DataFrame,
    software_counts: pd.DataFrame,
    software_percentages: pd.DataFrame,
    diagnosis_n: Dict[str, int],
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = GridSpec(1, 2, figure=figure, width_ratios=[1, 1.2])
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])

    tile_matrix(
        axis_a,
        feature_counts,
        feature_percentages,
        diagnosis_n,
        "A. Diagnosis × acoustic feature family",
    )
    tile_matrix(
        axis_b,
        software_counts,
        software_percentages,
        diagnosis_n,
        "B. Diagnosis × software / extraction environment",
    )

    figure.suptitle(
        "Diagnosis-specific feature and software usage",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    colour_bar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="Blues"),
        ax=[axis_a, axis_b],
        fraction=0.022,
        pad=0.025,
    )
    colour_bar.set_label("Studies within diagnosis (%)", fontsize=9)
    colour_bar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colour_bar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    colour_bar.ax.tick_params(labelsize=12)

    figure.text(
        0.01,
        -0.045,
        "Cell shade shows the percentage of studies within each diagnosis. "
        "Each cell reports n and percentage; * indicates fewer than 10 publications. "
        "Blank cells represent zero studies.",
        fontsize=15,
    )

    save_figure(figure, output_dir, "Figure_2_alternative_tile_matrices")


# -----------------------------------------------------------------------------
# Figure 3: grouped dot plot and repository score distributions
# -----------------------------------------------------------------------------
def create_figure3(
    reporting: pd.DataFrame,
    repository_domains: pd.DataFrame,
    n_repositories: int,
    output_dir: Path,
) -> None:
    positions = []
    labels = []
    bounds = []
    current_y = 0.0

    for domain in PIPELINE_ORDER:
        subset = reporting.loc[reporting["Domain"].eq(domain)]
        if subset.empty:
            continue
        start = current_y
        for _, row in subset.iterrows():
            positions.append(current_y)
            labels.append(FIGURE3A_SHORT_LABELS.get(row["Item"], row["Item"]))
            current_y += 1
        end = current_y - 1
        bounds.append((domain, start, end))
        current_y += 0.8

    repository_plot = repository_domains.copy()
    repository_plot["Score 0 percent"] = (
        repository_plot["Score 0"] / repository_plot["n scored"] * 100
    )
    repository_plot["Score 1 percent"] = (
        repository_plot["Score 1"] / repository_plot["n scored"] * 100
    )
    repository_plot["Score 2 percent"] = (
        repository_plot["Score 2"] / repository_plot["n scored"] * 100
    )
    repository_plot = repository_plot.sort_values(
        "Mean score", ascending=False
    ).reset_index(drop=True)

    figure = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = GridSpec(
        2,
        2,
        figure=figure,
        width_ratios=[1.05, 1.1],
        height_ratios=[1, 0.055],
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[1, 1])
    legend_axis.axis("off")

    band_colours = {
        "Input": "#F5F7FB",
        "Preprocessing": "#F7FBF5",
        "Extraction": "#FBF7F5",
        "Implementation": "#F9F5FB",
        "Validation": "#F5FAFB",
    }

    for domain, start, end in bounds:
        axis_a.add_patch(
            Rectangle(
                (0, start - 0.45),
                100,
                (end - start) + 0.9,
                facecolor=band_colours[domain],
                edgecolor="none",
                zorder=0,
            )
        )
        # Put the group name inside the plot rather than over the item labels.
        axis_a.text(
            98,
            start - 0.22,
            domain,
            va="bottom",
            ha="right",
            fontsize=15,
            fontweight="bold",
            color="#555555",
        )

    for position, (_, row) in zip(positions, reporting.iterrows()):
        percentage = float(row["Percent"]) * 100
        axis_a.hlines(
            position,
            0,
            percentage,
            color="#9E9E9E",
            linewidth=1.2,
            zorder=1,
        )
        axis_a.plot(
            percentage,
            position,
            "o",
            markersize=8,
            color="#2C7FB8",
            zorder=2,
        )
        axis_a.text(
            percentage + 1.2,
            position,
            f"{int(row['n'])} ({percentage:.1f}%)",
            va="center",
            ha="left",
            fontsize=12,
        )

    axis_a.set_yticks(positions)
    axis_a.set_yticklabels(labels, fontsize=12)
    axis_a.set_xlim(0, 112)
    axis_a.set_xlabel("Publications reporting item (%)", fontsize=12)
    axis_a.set_title(
        "A. Publication reporting completeness",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    axis_a.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    axis_a.set_axisbelow(True)
    axis_a.invert_yaxis()
    axis_a.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        axis_a.spines[spine].set_visible(False)

    y_positions = np.arange(len(repository_plot))
    axis_b.barh(
        y_positions,
        repository_plot["Score 0 percent"],
        color="#D9D9D9",
        edgecolor="white",
        label="0 = absent",
    )
    axis_b.barh(
        y_positions,
        repository_plot["Score 1 percent"],
        left=repository_plot["Score 0 percent"],
        color="#9ECAE1",
        edgecolor="white",
        label="1 = partial",
    )
    axis_b.barh(
        y_positions,
        repository_plot["Score 2 percent"],
        left=(
            repository_plot["Score 0 percent"]
            + repository_plot["Score 1 percent"]
        ),
        color="#3182BD",
        edgecolor="white",
        label="2 = complete",
    )

    for row_number, row in repository_plot.iterrows():
        axis_b.text(
            101.5,
            row_number,
            f"mean {row['Mean score']:.2f}/2",
            va="center",
            ha="left",
            fontsize=12,
        )

        left = 0.0
        for percentage_column, count_column in [
            ("Score 0 percent", "Score 0"),
            ("Score 1 percent", "Score 1"),
            ("Score 2 percent", "Score 2"),
        ]:
            segment_percentage = float(row[percentage_column])
            if segment_percentage >= 9:
                axis_b.text(
                    left + segment_percentage / 2,
                    row_number,
                    str(int(row[count_column])),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )
            left += segment_percentage

    axis_b.set_yticks(y_positions)
    axis_b.set_yticklabels(
        [FIGURE3B_SHORT_LABELS.get(label, label) for label in repository_plot["Domain"]],
        fontsize=12,
    )
    axis_b.set_xlim(0, 116)
    axis_b.set_xlabel("Repositories by score level (%)", fontsize=12)
    axis_b.set_title(
        "B. Repository quality-domain score distributions",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    # Keep the score legend in a dedicated row below Panel B, outside the plot.
    legend_handles, legend_labels = axis_b.get_legend_handles_labels()
    legend_axis.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=1.6,
        handlelength=1.8,
    )
    axis_b.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    axis_b.set_axisbelow(True)
    axis_b.invert_yaxis()
    axis_b.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        axis_b.spines[spine].set_visible(False)

    figure.suptitle(
        "Alternative Figure 3. Reproducibility reporting and repository quality",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    figure.text(
        0.01,
        -0.02,
        f"Panel A groups reporting items by the proposed pipeline stages. "
        f"Panel B shows scores across {n_repositories} complete repository appraisals.",
        fontsize=12,
    )

    save_figure(figure, output_dir, "Figure_3_alternative_reproducibility")


# -----------------------------------------------------------------------------
# Export and command-line entry point
# -----------------------------------------------------------------------------
def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for extension in ["png", "pdf", "svg"]:
        save_arguments = {"bbox_inches": "tight"}
        if extension == "png":
            save_arguments["dpi"] = 300
        figure.savefig(output_dir / f"{stem}.{extension}", **save_arguments)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing the publication CSV and GitHub XLSX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder in which figures and source-data CSVs will be written.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder does not exist:\n{data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    publications, quality, publication_file, github_file = load_inputs(data_dir)

    (
        feature_counts,
        feature_percentages,
        software_counts,
        software_percentages,
        diagnosis_n,
    ) = build_figure2_data(publications)

    reporting = build_reporting_data(len(publications))
    repository_domains, n_complete_repositories = build_repository_quality_data(
        quality
    )

    create_figure2(
        feature_counts,
        feature_percentages,
        software_counts,
        software_percentages,
        diagnosis_n,
        output_dir,
    )
    create_figure3(
        reporting,
        repository_domains,
        n_complete_repositories,
        output_dir,
    )

    # Export plotted values for checking and later editing.
    feature_counts.to_csv(output_dir / "Figure_2A_feature_counts.csv")
    feature_percentages.to_csv(
        output_dir / "Figure_2A_feature_percentages.csv"
    )
    software_counts.to_csv(output_dir / "Figure_2B_software_counts.csv")
    software_percentages.to_csv(
        output_dir / "Figure_2B_software_percentages.csv"
    )
    reporting.to_csv(
        output_dir / "Figure_3A_reporting_completeness.csv", index=False
    )
    repository_domains.to_csv(
        output_dir / "Figure_3B_repository_domain_scores.csv", index=False
    )

    print(f"Publication data: {publication_file}")
    print(f"GitHub data:      {github_file}")
    print(f"Publications:     {len(publications)}")
    print(f"Repositories:     {n_complete_repositories} complete appraisals")
    print(f"Outputs written:  {output_dir}")


if __name__ == "__main__":
    main()
