import argparse
import os
import sys
from pathlib import Path

import torch
# import wandb
from ultralytics import YOLO
# from wandb.integration.ultralytics import add_wandb_callback

os.environ["WANDB_MODE"] = "disabled"


def parse_arguments():

    '''args'''
    ### E:\Object_card>python src/train_tl.py --model D:\card_detection_proj_tashfiq\model\yolov8n-seg --data D:\card_detection_proj_tashfiq\data\data.yaml --epoch_stage1 2 --epoch_stage2 2

    #python src/train_tl.py --model E:/Object_card/model/yolov8n-seg --data E:/Object_card/data/data.yaml --epochs_stage1 2 --epochs_stage2 2
    parser = argparse.ArgumentParser(
        description="Card Detection Two-stage YOLOv8n-seg training."
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n-seg", help="YOLO model to load."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument(
        "--epochs_stage1", type=int, default=100, help="Stage 1: Frozen epochs"
    )
    parser.add_argument(
        "--epochs_stage2", type=int, default=100, help="Stage 2: Fine-tune epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--project",
        type=str,
        default="yolov8n_seg_twostage_card_detection",
        help="W&B project name",
    )
    parser.add_argument("--name", type=str, default="twostage_run", help="W&B run name")
    return parser.parse_args()


def train_stage(
    model,
    data_yaml,
    device,
    epochs,
    batch_size,
    imgsz,
    freeze,
    run_name,
    project,
    stage,
):
    model.train(
        data=str(data_yaml),
        device=device,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        name=f"{run_name}_stage{stage}",
        project=project,
        lr0=0.001 if stage == 1 else 0.0001,
        close_mosaic=10,
        cos_lr=True,
        cache=True,
        # fliplr=0.5,
        # flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        scale=0.4,
        save=True,
        save_period=10,
        exist_ok=True,
        freeze=freeze,
        patience=20 if stage == 2 else 0,
    )


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_yaml = Path(args.data)

    if not data_yaml.exists():
        print(f"data.yaml file not found: {data_yaml}")
        sys.exit(1)

    # Start W&B
    # wandb.login()
    # wandb.init(project=args.project, name=args.name, config=vars(args))

    # Load model
    model = YOLO(f"{args.model}.pt")
    # add_wandb_callback(model, enable_model_checkpointing=False)

    print("🔹 Starting Stage 1: Frozen training")
    train_stage(
        model,
        data_yaml,
        device,
        args.epochs_stage1,
        args.batch_size,
        args.imgsz,
        freeze=10,
        run_name=args.name,
        project=args.project,
        stage=1,
    )

    print("🔹 Starting Stage 2: Fine-tuning entire model")
    train_stage(
        model,
        data_yaml,
        device,
        args.epochs_stage2,
        args.batch_size,
        args.imgsz,
        freeze=0,
        run_name=args.name,
        project=args.project,
        stage=2,
    )

    # wandb.finish()


if __name__ == "__main__":
    main()
