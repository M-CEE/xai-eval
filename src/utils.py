"""
Shared utilities for the preprocessing pipeline: directory setup, a simple
file+console logger, and a figure-saving helper that follows the
filename-as-caption convention (no in-plot titles; captions recorded in a
sidecar text file).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def setup_output_dirs(dataset_name, domain, datasets_root="Datasets"):
    """Create (and return paths to) the standard output tree for one dataset,
    matching the repo layout:

        Datasets/<domain>/<dataset_name>/{raw, processed, metadata}

    Figures and logs are NOT separate top-level folders (to match the fixed
    3-folder-per-dataset convention); they nest inside processed/ and
    metadata/ respectively:

        raw/            -- the untouched, as-downloaded source file
        processed/data/  -- X_train.csv, X_test.csv, y_train.csv, y_test.csv
        processed/figures/ -- all saved plots + figure_captions.txt
        metadata/logs/   -- preprocessing_log.txt (narrative log)
        metadata/        -- dataset_info.json, preprocessing_log.json,
                             class_distribution.csv, feature_dictionary.csv
    """
    root = os.path.join(datasets_root, domain, dataset_name)
    paths = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "data": os.path.join(root, "processed", "data"),
        "figures": os.path.join(root, "processed", "figures"),
        "metadata": os.path.join(root, "metadata"),
        "logs": os.path.join(root, "metadata", "logs"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def setup_model_dirs(dataset_name, domain, model_name, models_root="Models"):
    """Create (and return paths to) the standard output tree for one
    (dataset x model) training run, parallel to the Datasets/ layout:

        Models/<domain>/<dataset_name>/<model_name>/{model, logs, figures, metadata}

        model/     -- serialized model artifact + feature schema
        logs/      -- narrative training log, full CV results
        figures/   -- performance plots, filename-as-caption
        metadata/  -- model_info.json, best_hyperparameters.json, test_metrics.json
    """
    root = os.path.join(models_root, domain, dataset_name, model_name)
    paths = {
        "root": root,
        "model": os.path.join(root, "model"),
        "logs": os.path.join(root, "logs"),
        "figures": os.path.join(root, "figures"),
        "metadata": os.path.join(root, "metadata"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


class Logger:
    """Minimal narrative logger: writes to logs/preprocessing_log.txt and stdout."""

    def __init__(self, log_dir, filename="preprocessing_log.txt"):
        self.path = os.path.join(log_dir, filename)
        with open(self.path, "w") as f:
            f.write("PREPROCESSING LOG\n")
            f.write("=" * 60 + "\n")

    def log(self, msg, also_print=True):
        with open(self.path, "a") as f:
            f.write(str(msg) + "\n")
        if also_print:
            print(msg)

    def section(self, title):
        self.log("\n" + "-" * 60)
        self.log(title)
        self.log("-" * 60)


def set_plot_style():
    """Apply a consistent, research-standard plotting style.

    Figures intentionally carry no in-plot title: the filename is the
    caption, and the full caption sentence is recorded separately via
    FigureRegistry below.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 120


class FigureRegistry:
    """Saves figures under descriptive filenames and records filename->caption
    mappings in figures/figure_captions.txt, without ever writing the caption
    onto the figure itself.
    """

    def __init__(self, fig_dir, logger=None):
        self.fig_dir = fig_dir
        self.logger = logger
        self.caption_path = os.path.join(fig_dir, "figure_captions.txt")
        with open(self.caption_path, "w") as f:
            f.write("FIGURE CAPTIONS (filename -> caption)\n")
            f.write("=" * 60 + "\n")

    def save(self, fig, filename, caption):
        path = os.path.join(self.fig_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        with open(self.caption_path, "a") as f:
            f.write(f"{filename}: {caption}\n")
        if self.logger:
            self.logger.log(f"[FIGURE SAVED] {filename}  |  caption: {caption}")
        return path
