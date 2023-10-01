import argparse
import string
from pathlib import Path
from shutil import copyfile

import imagesize
from tqdm import tqdm

NAMES = {
    "cd": {0: "plate"},
    "cr": {k: v for k, v in enumerate(string.ascii_uppercase + string.digits)},
}


def cxcwh_to_xyxy(bbox: list[float], W: int, H: int):
    """
    Convert bounding box coordinates from [center_x, center_y, width, height] to [x1, y1, x2, y2].

    Parameters
    ----------
    bbox : list of floats
        The bounding box coordinates in the format [center_x, center_y, width, height].
    W : int
        The width of the image.
    H : int
        The height of the image.

    Returns
    -------
    list of floats
        The converted bounding box coordinates in the format [x1, y1, x2, y2].

    """

    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H

    return [x1, y1, x2, y2]


def rescale_images(src_folder: Path, out_folder: Path, task: str):
    """
    Create a new folder containing rescaled images and corresponding annotations.

    Parameters
    ----------
    src_folder : Path
        Path to the folder containing the original images and annotations.
    out_folder : Path
        Path to the folder where rescaled images and annotations will be saved.
    task : str
        The task to perform, either "cd" (code detection) or "cr" (code recognition).
    """

    images = list(src_folder.rglob("*.jpg")) + list(src_folder.rglob("*.jpeg"))

    for image_path in tqdm(images, desc=f"Rescaling {task} images"):
        w, h = imagesize.get(image_path)

        annot = []
        txt_path = image_path.with_suffix(".txt")

        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").split(" ")
                cxcywh = [float(x) for x in line[1:]]
                print(cxcywh)
                xyxy = cxcwh_to_xyxy(cxcywh, w, h)

                cls_idx = int(line[0])
                cls_name = NAMES[task][cls_idx]

                annot.append([cls_name, cls_idx] + xyxy)

        annot.sort(key=lambda x: x[2])
        annot_str = ""

        for line in annot:
            annot_str += (
                f"{w},{h} {line[0]} {line[1]} {line[2]} {line[3]} {line[4]} {line[5]}\n"
            )

        if task == "cd":
            copyfile(image_path, out_folder / image_path.name)

        new_txt_id = image_path.stem + f"_{task}.txt"
        with open(out_folder / new_txt_id, "w") as f:
            f.write(annot_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cd-folder", type=str)
    parser.add_argument("--cr-folder", type=str)
    args = parser.parse_args()

    cd_folder = Path(args.cd_folder).absolute()
    cr_folder = Path(args.cr_folder).absolute()

    out_dir = cd_folder.parent / "rescaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    rescale_images(cd_folder, out_dir, "cd")
    rescale_images(cr_folder, out_dir, "cr")
