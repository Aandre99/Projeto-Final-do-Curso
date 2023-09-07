import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str)
    args = parser.parse_args()

    plot_results(Path(args.images).absolute())
