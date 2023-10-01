import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_folder: Path):

    output_folder = video_folder / "frames"
    videos_path = list(Path(video_folder).glob("*"))

    for video_path in tqdm(videos_path, desc="Extracting frames", total=len(list(videos_path))):
        
        output_folder_i = output_folder / video_path.stem 
        output_folder_i.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = str(output_folder_i / f"{video_path.stem}_{frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

            frame_number += 1

        cap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    
    parser.add_argument("--video-folder", help="Path to the video file")
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    extract_frames(video_folder)