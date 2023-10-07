import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_accuracy_by_confidence(
    df: pd.DataFrame, conf_thresholds: list[float]
) -> list[float]:

    """
    This function returns the accuracy of the model for each confidence threshold

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the model predictions
    conf_thresholds : list[float]
        The list of confidence thresholds to use

    Returns
    -------
    list[float]
        The list of accuracies for each confidence threshold
    """

    accs = []

    for conf_threshold in conf_thresholds:

        filter_str = f"conf_thresh == {conf_threshold}"
        filtered_df = df.query(filter_str)
        accs.append(
            filtered_df.query("pred_thresh == label").shape[0] / filtered_df.shape[0]
        )

    return accs


def plot_desktop_accuracy(
    pt_accs: list[float],
    tf_accs: list[float],
    conf_thresholds: list[float],
    figsize=(10, 5),
    output_dir: Path = "",
):

    """
    This function plots the accuracy of the model for each confidence threshold

    Parameters
    ----------
    pt_accs : list[float]
        The list of accuracies for each confidence threshold for PyTorch

    tf_accs : list[float]
        The list of accuracies for each confidence threshold for TFLite

    conf_thresholds : list[float]
        The list of confidence thresholds to use

    """

    plt.figure(figsize=figsize)

    plt.plot(
        conf_thresholds,
        pt_accs,
        label="PyTorch Model ALPR",
        color="blue",
        marker="o",
    )
    for i, txt in enumerate(pt_accs):
        plt.text(
            conf_thresholds[i],
            pt_accs[i] - 0.02,
            f"{txt:.3f}",
            ha="center",
            va="top",
            fontsize=10,
            color="blue",
        )

    plt.plot(
        conf_thresholds,
        tf_accs,
        label="TFLite   Model ALPR",
        color="orange",
        marker="*",
        linestyle="--",
    )
    for i, txt in enumerate(tf_accs):
        plt.text(
            conf_thresholds[i],
            tf_accs[i] + 0.02,
            f"{txt:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="orange",
        )

    plt.ylabel("Accuracy")
    plt.xlabel("Confidence Threshold")
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xticks(conf_thresholds)
    plt.title("ALPR Performance")
    plt.legend(loc="lower left")
    plt.savefig(output_dir / "accuracy.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Results Graphs")
    parser.add_argument("--pt-output-csv", type=Path, required=True)
    parser.add_argument("--tf-output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pt_df = pd.read_csv(args.pt_output_csv)
    tf_df = pd.read_csv(args.tf_output_csv)

    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pt_accs = get_accuracy_by_confidence(pt_df, conf_thresholds)
    tf_accs = get_accuracy_by_confidence(tf_df, conf_thresholds)

    plot_desktop_accuracy(
        pt_accs,
        tf_accs,
        conf_thresholds,
        output_dir=args.output_dir,
    )
