import argparse
from pathlib import Path

import pandas as pd

from utils.data import *
from utils.metrics import build_accuracy_df, build_metrics_df, get_metrics
from utils.plot import plot_boxes, plot_two_boxes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mobile OCR Benchmark")
    parser.add_argument("--images-dir", type=Path, help="Path to images directory")
    parser.add_argument("--labels-dir", type=Path, help="Path to labels directory")
    parser.add_argument("--output-dir", type=Path, help="Path to save results files")
    parser.add_argument("--jsons-dir", type=Path, help="Path to jsons files")
    args = parser.parse_args()

    data = {}
    metrics = {}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for device_name in ["GPU", "CPU"]:

        print(f"\n> Running Mobile ALPR Benchmark with {device_name} predictions\n")

        print(f"\n[+] Load Json file with predictions")
        json_path = args.jsons_dir / f"detections{device_name}.json"
        benchmark_dict = json_to_dict(json_path)

        print(f"[+] Build predictions dictionary")
        predictions = get_predictions(args.labels_dir, benchmark_dict)

        print(f"[+] Build groundtruths dictionary")
        groundtruths = get_groundtruth(args.images_dir, args.labels_dir)

        print(f"[+] Fit predictions to groundtruths")
        predictions = fit_pred_boxes_to_gt_boxes(groundtruths, predictions)

        # plot_two_boxes(groundtruths,predictions, args.images_dir, n_samples=5)
        print('[+] Computing metrics')

        device_name_data, device_name_metrics = get_metrics(groundtruths, predictions)
        data[device_name] = device_name_data
        metrics[device_name] = device_name_metrics

    build_accuracy_df(data['GPU'], data['CPU'], args.output_dir)
    build_metrics_df(metrics["GPU"], metrics["CPU"], args.output_dir)
