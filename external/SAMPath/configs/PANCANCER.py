from box import Box

config = {
    "batch_size": 1,
    "num_workers": 0,
    "out_dir": "./outputs/PANCANCER",
    "opt": {
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "precision": 32,
        "steps": [1],
        "warmup_steps": 0,
    },
    "model": {
        "type": "vit_b",
        "checkpoint": "./weights/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,
        "extra_encoder": None,
        "extra_type": "plus",
    },
    "loss": {
        "focal_cof": 0.25,
        "dice_cof": 0.75,
        "ce_cof": 0.0,
        "iou_cof": 0.0,
    },
    "dataset": {
        "dataset_root": "./dataset/pancancer",
        "dataset_csv_path": "./dataset_cfg/pancancer.csv",
        "data_ext": ".png",
        "val_fold_id": 0,
        "num_classes": 6,
        "ignored_classes": 0,
        "ignored_classes_metric": None,
        "image_hw": (1024, 1024),
        "feature_input": False,
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std": (0.229, 0.224, 0.225),
    },
}

cfg = Box(config)
