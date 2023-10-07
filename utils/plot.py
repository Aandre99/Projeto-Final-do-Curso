import argparse
from pathlib import Path

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")


def plot_results(images_folder: Path):
    """
    @brief plots boxes of the detected plates and characters on the images

    :param images_folder: folder containing the images and the corresponding annotations

    @return: None
    """

    paths = list(images_folder.glob("*.jpg"))[:5]

    for image_path in paths:
        print(image_path.name)
        image2plot = cv.imread(str(image_path))
        image2plot = cv.cvtColor(image2plot, cv.COLOR_BGR2RGB)

        txt_cr_path = images_folder / (image_path.stem + "_cr.txt")
        txt_cd_path = images_folder / (image_path.stem + "_cd.txt")

        patch2plot = None

        with open(txt_cd_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").split(" ")[3:]
                [x1, y1, x2, y2] = [float(x) for x in line]

                patch2plot = image2plot[int(y1) : int(y2), int(x1) : int(x2)].copy()
                image2plot = cv.rectangle(
                    image2plot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3
                )

        with open(txt_cr_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").split(" ")[3:]
                x1, y1, x2, y2 = [float(x) for x in line]

                patch2plot = cv.rectangle(
                    patch2plot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1
                )

        plt.subplot(1, 2, 1)
        plt.imshow(image2plot)
        plt.subplot(1, 2, 2)
        plt.imshow(patch2plot)
        plt.show()


def plot_boxes(gt_samples: dict, samples: dict, images_dir: Path, n_samples: int = 10):

    for i, (image_id, image_info) in enumerate(samples.items()):

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        if i == n_samples:
            break

        image2plot = cv.imread(str(images_dir / image_id))
        image2plot = cv.cvtColor(image2plot, cv.COLOR_BGR2RGB)

        plate2plot = None

        for box in gt_samples[image_id]["cd_boxes"]:
            x1, y1, x2, y2 = box[2:]

            plate2plot = image2plot[int(y1) : int(y2), int(x1) : int(x2)].copy()

            image2plot = cv.rectangle(
                image2plot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3
            )

        for box in image_info["cr_boxes"]:
            x1, y1, x2, y2 = box[2:]

            plate2plot = cv.rectangle(
                plate2plot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1
            )

        plt.subplot(1, 2, 1)
        plt.imshow(image2plot)

        plt.subplot(1, 2, 2)
        plt.imshow(plate2plot)
        plt.title(f"plate shape = {plate2plot.shape}")

        plt.show()
        print(image_id)


def plot_two_boxes(gts: dict, preds: dict, images_dir: Path, n_samples: int = -1):

    for i, (image_id, image_info) in enumerate(preds.items()):

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        # if i == n_samples:
        #     break

        image2plot = cv.imread(str(images_dir / image_id))
        image2plot = cv.cvtColor(image2plot, cv.COLOR_BGR2RGB)

        plate2plot = None

        gt_cd_boxes = gts[image_id]["cd_boxes"]
        gt_cr_boxes = gts[image_id]["cr_boxes"]

        pred_cd_boxes = image_info["cd_boxes"]
        pred_cr_boxes = image_info["cr_boxes"]

        for pred_box, gt_box in zip(pred_cd_boxes, gt_cd_boxes):

            x1_pred, y1_pred, x2_pred, y2_pred = pred_box[2:]
            x1_gt, y1_gt, x2_gt, y2_gt = gt_box[2:]

            plate2plot = image2plot[
                int(y1_gt) : int(y2_gt), int(x1_gt) : int(x2_gt)
            ].copy()

            image2plot = cv.rectangle(
                image2plot,
                (int(x1_pred), int(y1_pred)),
                (int(x2_pred), int(y2_pred)),
                (255, 0, 0),
                3,
            )

            image2plot = cv.rectangle(
                image2plot,
                (int(x1_gt), int(y1_gt)),
                (int(x2_gt), int(y2_gt)),
                (0, 255, 0),
                3,
            )

        for pred_box, gt_box in zip(pred_cr_boxes, gt_cr_boxes):

            x1_pred, y1_pred, x2_pred, y2_pred = pred_box[2:]
            x1_gt, y1_gt, x2_gt, y2_gt = gt_box[2:]

            plate2plot = cv.rectangle(
                plate2plot,
                (int(x1_pred), int(y1_pred)),
                (int(x2_pred), int(y2_pred)),
                (255, 0, 0),
                3,
            )

            plate2plot = cv.rectangle(
                plate2plot,
                (int(x1_gt), int(y1_gt)),
                (int(x2_gt), int(y2_gt)),
                (0, 255, 0),
                3,
            )

        plt.subplot(1, 2, 1)
        plt.imshow(image2plot)

        plt.subplot(1, 2, 2)
        plt.imshow(plate2plot)

        plt.show()


def plot_desktop_benchmark_metrics(output_folder: Path):

    df_tf = pd.read_csv(output_folder.absolute() / "tflite" / "data.csv")
    df_pt = pd.read_csv(output_folder.absolute() / "pt" / "data.csv")

    tf_accs = []
    pt_accs = []

    tf_undets = df_tf.query("conf_thresh == -1").shape[0]
    pt_undets = df_pt.query("conf_thresh == -1").shape[0]

    confs = np.arange(0.1, 1.0, 0.1)

    for conf in confs:

        tf_conf = df_tf.query(f"conf_thresh >= {conf}")
        pt_conf = df_pt.query(f"conf_thresh >= {conf}")

        tf_accs.append(
            tf_conf.query("label == pred_thresh").shape[0]
            / (tf_conf.shape[0] + tf_undets)
        )

        pt_accs.append(
            pt_conf.query("label == pred_thresh").shape[0]
            / (pt_conf.shape[0] + pt_undets)
        )

    print(tf_accs)
    print(pt_accs)

    plt.plot(confs, tf_accs, label="TensorFlow", color="orange", linestyle="-")
    plt.plot(confs, pt_accs, label="PyTorch", color="blue", linestyle="--")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("OCR Accuracy")
    plt.xticks(confs)
    plt.yticks(confs)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str)
    args = parser.parse_args()

    # plot_results(Path(args.images).absolute())

    plot_desktop_benchmark_metrics(Path(r"C:\Users\santo\dev\TCC\data\and_bench\outs"))
