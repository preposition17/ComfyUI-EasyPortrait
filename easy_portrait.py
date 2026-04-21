from __future__ import annotations

import copy
import os
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import requests
import torch
from PIL import Image

import folder_paths

MODEL_ROOT = os.path.join(folder_paths.models_dir, "easyportrait")
CHECKPOINT_ROOT = os.path.join(MODEL_ROOT, "checkpoints")
ONNX_ROOT = os.path.join(MODEL_ROOT, "onnx")
ONNX_REPO_ID = "sadzip/EasyPortrait-ONNX"
ONNX_REPO_URL = f"https://huggingface.co/{ONNX_REPO_ID}/resolve/main"
folder_paths.add_model_folder_path("easyportrait", MODEL_ROOT)

IMG_MEAN = [143.55267075, 132.96705975, 126.94924335]
IMG_STD = [60.2625333, 60.32740275, 59.30988645]

TASK_PORTRAIT = "portrait_segmentation"
TASK_FACE = "face_parsing"

CLASS_NAMES = {
    TASK_PORTRAIT: ("background", "person"),
    TASK_FACE: (
        "background",
        "skin",
        "left brow",
        "right brow",
        "left eye",
        "right eye",
        "lips",
        "teeth",
    ),
}

TASK_COLORS = {
    TASK_PORTRAIT: np.array([0, 255, 127], dtype=np.uint8),
    TASK_FACE: np.array([255, 140, 0], dtype=np.uint8),
}

LABEL_COLORS = {
    "background": np.array([32, 32, 32], dtype=np.uint8),
    "person": np.array([0, 255, 127], dtype=np.uint8),
    "skin": np.array([255, 184, 128], dtype=np.uint8),
    "left brow": np.array([128, 80, 255], dtype=np.uint8),
    "right brow": np.array([88, 120, 255], dtype=np.uint8),
    "left eye": np.array([0, 220, 255], dtype=np.uint8),
    "right eye": np.array([0, 148, 255], dtype=np.uint8),
    "lips": np.array([255, 72, 128], dtype=np.uint8),
    "teeth": np.array([245, 245, 220], dtype=np.uint8),
}

MASK_MODES = ["binary", "layers"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    checkpoint_url: str
    checkpoint_filename: str
    task: str
    input_size: int
    builder_name: str | None


def _make_checkpoint_url(name: str) -> str:
    return (
        "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
        f"datasets/easyportrait/experiments/models/{name}"
    )


MODEL_SPECS = [
    ModelSpec(
        "bisenet_ps",
        "Portrait Segmentation / BiSeNet-V2 / 384",
        _make_checkpoint_url("bisenet-ps.pth"),
        "bisenet-ps.pth",
        TASK_PORTRAIT,
        384,
        "bisenet",
    ),
    ModelSpec(
        "danet_ps",
        "Portrait Segmentation / DANet / 384",
        _make_checkpoint_url("danet-ps.pth"),
        "danet-ps.pth",
        TASK_PORTRAIT,
        384,
        "danet",
    ),
    ModelSpec(
        "deeplabv3_ps",
        "Portrait Segmentation / DeepLabv3 / 384",
        _make_checkpoint_url("deeplabv3-ps.pth"),
        "deeplabv3-ps.pth",
        TASK_PORTRAIT,
        384,
        "deeplabv3",
    ),
    ModelSpec(
        "extremec3net_ps",
        "Portrait Segmentation / ExtremeC3Net / 384",
        _make_checkpoint_url("extremenet-ps.pth"),
        "extremenet-ps.pth",
        TASK_PORTRAIT,
        384,
        None,
    ),
    ModelSpec(
        "fastscnn_ps",
        "Portrait Segmentation / Fast SCNN / 384",
        _make_checkpoint_url("fast_scnn-ps.pth"),
        "fast_scnn-ps.pth",
        TASK_PORTRAIT,
        384,
        "fastscnn",
    ),
    ModelSpec(
        "fcn_mobilenetv2_ps",
        "Portrait Segmentation / FCN + MobileNetv2 / 384",
        _make_checkpoint_url("fcn-ps.pth"),
        "fcn-ps.pth",
        TASK_PORTRAIT,
        384,
        "fcn_mobilenetv2",
    ),
    ModelSpec(
        "fpn_resnet50_ps_1024",
        "Portrait Segmentation / FPN + ResNet50 / 1024",
        _make_checkpoint_url("fpn-ps-1024.pth"),
        "fpn-ps-1024.pth",
        TASK_PORTRAIT,
        1024,
        "fpn_resnet50",
    ),
    ModelSpec(
        "fpn_resnet50_ps_512",
        "Portrait Segmentation / FPN + ResNet50 / 512",
        _make_checkpoint_url("fpn-ps-512.pth"),
        "fpn-ps-512.pth",
        TASK_PORTRAIT,
        512,
        "fpn_resnet50",
    ),
    ModelSpec(
        "fpn_resnet50_ps_384",
        "Portrait Segmentation / FPN + ResNet50 / 384",
        _make_checkpoint_url("fpn-ps.pth"),
        "fpn-ps.pth",
        TASK_PORTRAIT,
        384,
        "fpn_resnet50",
    ),
    ModelSpec(
        "fpn_resnet50_ps_224",
        "Portrait Segmentation / FPN + ResNet50 / 224",
        _make_checkpoint_url("fpn-ps-224.pth"),
        "fpn-ps-224.pth",
        TASK_PORTRAIT,
        224,
        "fpn_resnet50",
    ),
    ModelSpec(
        "segformer_b0_ps_1024",
        "Portrait Segmentation / SegFormer-B0 / 1024",
        _make_checkpoint_url("segformer-ps-1024.pth"),
        "segformer-ps-1024.pth",
        TASK_PORTRAIT,
        1024,
        "segformer_b0",
    ),
    ModelSpec(
        "segformer_b0_ps_512",
        "Portrait Segmentation / SegFormer-B0 / 512",
        _make_checkpoint_url("segformer-ps-512.pth"),
        "segformer-ps-512.pth",
        TASK_PORTRAIT,
        512,
        "segformer_b0",
    ),
    ModelSpec(
        "segformer_b0_ps_384",
        "Portrait Segmentation / SegFormer-B0 / 384",
        _make_checkpoint_url("segformer-ps.pth"),
        "segformer-ps.pth",
        TASK_PORTRAIT,
        384,
        "segformer_b0",
    ),
    ModelSpec(
        "segformer_b0_ps_224",
        "Portrait Segmentation / SegFormer-B0 / 224",
        _make_checkpoint_url("segformer-ps-224.pth"),
        "segformer-ps-224.pth",
        TASK_PORTRAIT,
        224,
        "segformer_b0",
    ),
    ModelSpec(
        "sinet_ps",
        "Portrait Segmentation / SINet / 384",
        _make_checkpoint_url("sinet-ps.pth"),
        "sinet-ps.pth",
        TASK_PORTRAIT,
        384,
        None,
    ),
    ModelSpec(
        "bisenet_fp",
        "Face Parsing / BiSeNet-V2 / 384",
        _make_checkpoint_url("bisenet-fp.pth"),
        "bisenet-fp.pth",
        TASK_FACE,
        384,
        "bisenet",
    ),
    ModelSpec(
        "danet_fp",
        "Face Parsing / DANet / 384",
        _make_checkpoint_url("danet-fp.pth"),
        "danet-fp.pth",
        TASK_FACE,
        384,
        "danet",
    ),
    ModelSpec(
        "deeplabv3_fp",
        "Face Parsing / DeepLabv3 / 384",
        _make_checkpoint_url("deeplabv3-fp.pth"),
        "deeplabv3-fp.pth",
        TASK_FACE,
        384,
        "deeplabv3",
    ),
    ModelSpec(
        "ehanet_fp",
        "Face Parsing / EHANet / 384",
        _make_checkpoint_url("ehanet-fp.pth"),
        "ehanet-fp.pth",
        TASK_FACE,
        384,
        None,
    ),
    ModelSpec(
        "fastscnn_fp",
        "Face Parsing / Fast SCNN / 384",
        _make_checkpoint_url("fast_scnn-fp.pth"),
        "fast_scnn-fp.pth",
        TASK_FACE,
        384,
        "fastscnn",
    ),
    ModelSpec(
        "fcn_mobilenetv2_fp",
        "Face Parsing / FCN + MobileNetv2 / 384",
        _make_checkpoint_url("fcn-fp.pth"),
        "fcn-fp.pth",
        TASK_FACE,
        384,
        "fcn_mobilenetv2",
    ),
    ModelSpec(
        "fpn_resnet50_fp_1024",
        "Face Parsing / FPN + ResNet50 / 1024",
        _make_checkpoint_url("fpn-fp-1024.pth"),
        "fpn-fp-1024.pth",
        TASK_FACE,
        1024,
        "fpn_resnet50",
    ),
    ModelSpec(
        "fpn_resnet50_fp_512",
        "Face Parsing / FPN + ResNet50 / 512",
        _make_checkpoint_url("fpn-fp-512.pth"),
        "fpn-fp-512.pth",
        TASK_FACE,
        512,
        "fpn_resnet50",
    ),
    ModelSpec(
        "fpn_resnet50_fp_384",
        "Face Parsing / FPN + ResNet50 / 384",
        _make_checkpoint_url("fpn-fp.pth"),
        "fpn-fp.pth",
        TASK_FACE,
        384,
        "fpn_resnet50",
    ),
    ModelSpec(
        "fpn_resnet50_fp_224",
        "Face Parsing / FPN + ResNet50 / 224",
        _make_checkpoint_url("fpn-fp-224.pth"),
        "fpn-fp-224.pth",
        TASK_FACE,
        224,
        "fpn_resnet50",
    ),
    ModelSpec(
        "segformer_b0_fp_1024",
        "Face Parsing / SegFormer-B0 / 1024",
        _make_checkpoint_url("segformer-fp-1024.pth"),
        "segformer-fp-1024.pth",
        TASK_FACE,
        1024,
        "segformer_b0",
    ),
    ModelSpec(
        "segformer_b0_fp_512",
        "Face Parsing / SegFormer-B0 / 512",
        _make_checkpoint_url("segformer-fp-512.pth"),
        "segformer-fp-512.pth",
        TASK_FACE,
        512,
        "segformer_b0",
    ),
    ModelSpec(
        "segformer_b0_fp_384",
        "Face Parsing / SegFormer-B0 / 384",
        _make_checkpoint_url("segformer-fp.pth"),
        "segformer-fp.pth",
        TASK_FACE,
        384,
        "segformer_b0",
    ),
    ModelSpec(
        "segformer_b0_fp_224",
        "Face Parsing / SegFormer-B0 / 224",
        _make_checkpoint_url("segformer-fp-224.pth"),
        "segformer-fp-224.pth",
        TASK_FACE,
        224,
        "segformer_b0",
    ),
]

ONNX_MODEL_SPECS = [spec for spec in MODEL_SPECS if spec.builder_name is not None]
MODEL_CHOICES = [spec.display_name for spec in ONNX_MODEL_SPECS]
MODELS_BY_NAME = {spec.display_name: spec for spec in ONNX_MODEL_SPECS}


def _num_classes(task: str) -> int:
    return 2 if task == TASK_PORTRAIT else 8


def _norm_cfg(momentum: float | None = None) -> dict:
    cfg = {"type": "SyncBN", "requires_grad": True}
    if momentum is not None:
        cfg["momentum"] = momentum
    return cfg


def _test_pipeline(image_size: int) -> list[dict]:
    return [
        {"type": "LoadImageFromFile"},
        {
            "type": "MultiScaleFlipAug",
            "img_scale": (image_size, image_size),
            "flip": False,
            "transforms": [
                {
                    "type": "Normalize",
                    "mean": IMG_MEAN,
                    "std": IMG_STD,
                    "to_rgb": True,
                },
                {"type": "ImageToTensor", "keys": ["img"]},
                {"type": "Collect", "keys": ["img"]},
            ],
        },
    ]


def _base_config(spec: ModelSpec, model_dict: dict) -> dict:
    return {
        "model": model_dict,
        "test_cfg": {"mode": "whole"},
        "test_pipeline": _test_pipeline(spec.input_size),
        "cudnn_benchmark": True,
    }


def _build_bisenet(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "pretrained": None,
            "backbone": {
                "type": "BiSeNetV2",
                "detail_channels": (64, 64, 128),
                "semantic_channels": (16, 32, 64, 128),
                "semantic_expansion_ratio": 6,
                "bga_channels": 128,
                "out_indices": (0, 1, 2, 3, 4),
                "init_cfg": None,
                "align_corners": False,
            },
            "decode_head": {
                "type": "FCNHead",
                "in_channels": 128,
                "in_index": 0,
                "channels": 1024,
                "num_convs": 1,
                "concat_input": False,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 1.0,
                },
                "sampler": {
                    "type": "OHEMPixelSampler",
                    "thresh": 0.7,
                    "min_kept": 10000,
                },
            },
            "auxiliary_head": [
                {
                    "type": "FCNHead",
                    "in_channels": 16,
                    "channels": 16,
                    "num_convs": 2,
                    "num_classes": num_classes,
                    "in_index": 1,
                    "norm_cfg": _norm_cfg(),
                    "concat_input": False,
                    "align_corners": False,
                    "sampler": {
                        "type": "OHEMPixelSampler",
                        "thresh": 0.7,
                        "min_kept": 10000,
                    },
                    "loss_decode": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                },
                {
                    "type": "FCNHead",
                    "in_channels": 32,
                    "channels": 64,
                    "num_convs": 2,
                    "num_classes": num_classes,
                    "in_index": 2,
                    "norm_cfg": _norm_cfg(),
                    "concat_input": False,
                    "align_corners": False,
                    "sampler": {
                        "type": "OHEMPixelSampler",
                        "thresh": 0.7,
                        "min_kept": 10000,
                    },
                    "loss_decode": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                },
                {
                    "type": "FCNHead",
                    "in_channels": 64,
                    "channels": 256,
                    "num_convs": 2,
                    "num_classes": num_classes,
                    "in_index": 3,
                    "norm_cfg": _norm_cfg(),
                    "concat_input": False,
                    "align_corners": False,
                    "sampler": {
                        "type": "OHEMPixelSampler",
                        "thresh": 0.7,
                        "min_kept": 10000,
                    },
                    "loss_decode": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                },
                {
                    "type": "FCNHead",
                    "in_channels": 128,
                    "channels": 1024,
                    "num_convs": 2,
                    "num_classes": num_classes,
                    "in_index": 4,
                    "norm_cfg": _norm_cfg(),
                    "concat_input": False,
                    "align_corners": False,
                    "sampler": {
                        "type": "OHEMPixelSampler",
                        "thresh": 0.7,
                        "min_kept": 10000,
                    },
                    "loss_decode": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                },
            ],
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


def _build_danet(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "pretrained": None,
            "backbone": {
                "type": "ResNetV1c",
                "depth": 50,
                "num_stages": 4,
                "out_indices": (0, 1, 2, 3),
                "dilations": (1, 1, 2, 4),
                "strides": (1, 2, 1, 1),
                "norm_cfg": _norm_cfg(),
                "norm_eval": False,
                "style": "pytorch",
                "contract_dilation": True,
            },
            "decode_head": {
                "type": "DAHead",
                "in_channels": 2048,
                "in_index": 3,
                "channels": 512,
                "pam_channels": 64,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 1.0,
                },
            },
            "auxiliary_head": {
                "type": "FCNHead",
                "in_channels": 1024,
                "in_index": 2,
                "channels": 256,
                "num_convs": 1,
                "concat_input": False,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 0.4,
                },
            },
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


def _build_deeplabv3(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "pretrained": None,
            "backbone": {
                "type": "ResNetV1c",
                "depth": 50,
                "num_stages": 4,
                "out_indices": (0, 1, 2, 3),
                "dilations": (1, 1, 2, 4),
                "strides": (1, 2, 1, 1),
                "norm_cfg": _norm_cfg(),
                "norm_eval": False,
                "style": "pytorch",
                "contract_dilation": True,
            },
            "decode_head": {
                "type": "ASPPHead",
                "in_channels": 2048,
                "in_index": 3,
                "channels": 512,
                "dilations": (1, 12, 24, 36),
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 1.0,
                },
            },
            "auxiliary_head": {
                "type": "FCNHead",
                "in_channels": 1024,
                "in_index": 2,
                "channels": 256,
                "num_convs": 1,
                "concat_input": False,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 0.4,
                },
            },
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


def _build_fastscnn(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "backbone": {
                "type": "FastSCNN",
                "downsample_dw_channels": (32, 48),
                "global_in_channels": 64,
                "global_block_channels": (64, 96, 128),
                "global_block_strides": (2, 2, 1),
                "global_out_channels": 128,
                "higher_in_channels": 64,
                "lower_in_channels": 128,
                "fusion_out_channels": 128,
                "out_indices": (0, 1, 2),
                "norm_cfg": _norm_cfg(momentum=0.01),
                "align_corners": False,
            },
            "decode_head": {
                "type": "DepthwiseSeparableFCNHead",
                "in_channels": 128,
                "channels": 128,
                "concat_input": False,
                "num_classes": num_classes,
                "in_index": -1,
                "norm_cfg": _norm_cfg(momentum=0.01),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": True,
                    "loss_weight": 1,
                },
            },
            "auxiliary_head": [
                {
                    "type": "FCNHead",
                    "in_channels": 128,
                    "channels": 32,
                    "num_classes": num_classes,
                },
                {
                    "type": "FCNHead",
                    "in_channels": 128,
                    "channels": 32,
                    "num_classes": num_classes,
                },
            ],
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


def _build_fcn_mobilenetv2(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "pretrained": None,
            "backbone": {
                "type": "MobileNetV2",
                "widen_factor": 1.0,
                "strides": (1, 2, 2, 1, 1, 1, 1),
                "dilations": (1, 1, 1, 2, 2, 4, 4),
                "out_indices": (1, 2, 4, 6),
                "norm_cfg": _norm_cfg(),
            },
            "decode_head": {
                "type": "FCNHead",
                "in_channels": 320,
                "in_index": 3,
                "channels": 512,
                "num_convs": 2,
                "concat_input": True,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 1.0,
                },
            },
            "auxiliary_head": {
                "type": "FCNHead",
                "in_channels": 96,
                "in_index": 2,
                "channels": 256,
                "num_convs": 1,
                "concat_input": False,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 0.4,
                },
            },
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


def _build_fpn_resnet50(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "pretrained": None,
            "backbone": {
                "type": "ResNetV1c",
                "depth": 50,
                "num_stages": 4,
                "out_indices": (0, 1, 2, 3),
                "dilations": (1, 1, 1, 1),
                "strides": (1, 2, 2, 2),
                "norm_cfg": _norm_cfg(),
                "norm_eval": False,
                "style": "pytorch",
                "contract_dilation": True,
            },
            "neck": {
                "type": "FPN",
                "in_channels": [256, 512, 1024, 2048],
                "out_channels": 256,
                "num_outs": 4,
            },
            "decode_head": {
                "type": "FPNHead",
                "in_channels": [256, 256, 256, 256],
                "in_index": [0, 1, 2, 3],
                "feature_strides": [4, 8, 16, 32],
                "channels": 128,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 1.0,
                },
            },
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


def _build_segformer_b0(spec: ModelSpec) -> dict:
    num_classes = _num_classes(spec.task)
    return _base_config(
        spec,
        {
            "type": "EncoderDecoder",
            "pretrained": None,
            "backbone": {
                "type": "MixVisionTransformer",
                "in_channels": 3,
                "embed_dims": 32,
                "num_stages": 4,
                "num_layers": [2, 2, 2, 2],
                "num_heads": [1, 2, 5, 8],
                "patch_sizes": [7, 3, 3, 3],
                "sr_ratios": [8, 4, 2, 1],
                "out_indices": (0, 1, 2, 3),
                "mlp_ratio": 4,
                "qkv_bias": True,
                "drop_rate": 0.0,
                "attn_drop_rate": 0.0,
                "drop_path_rate": 0.1,
            },
            "decode_head": {
                "type": "SegformerHead",
                "in_channels": [32, 64, 160, 256],
                "in_index": [0, 1, 2, 3],
                "channels": 256,
                "dropout_ratio": 0.1,
                "num_classes": num_classes,
                "norm_cfg": _norm_cfg(),
                "align_corners": False,
                "loss_decode": {
                    "type": "CrossEntropyLoss",
                    "use_sigmoid": False,
                    "loss_weight": 1.0,
                },
            },
            "train_cfg": {},
            "test_cfg": {"mode": "whole"},
        },
    )


CONFIG_BUILDERS: dict[str, Callable[[ModelSpec], dict]] = {
    "bisenet": _build_bisenet,
    "danet": _build_danet,
    "deeplabv3": _build_deeplabv3,
    "fastscnn": _build_fastscnn,
    "fcn_mobilenetv2": _build_fcn_mobilenetv2,
    "fpn_resnet50": _build_fpn_resnet50,
    "segformer_b0": _build_segformer_b0,
}


def _replace_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(module, torch.nn.SyncBatchNorm):
        converted = torch.nn.BatchNorm2d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                converted.weight.copy_(module.weight)
                converted.bias.copy_(module.bias)
        converted.running_mean = module.running_mean
        converted.running_var = module.running_var
        converted.num_batches_tracked = module.num_batches_tracked
        converted.training = module.training
        return converted

    for name, child in module.named_children():
        module.add_module(name, _replace_sync_batchnorm(child))
    return module


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _checkpoint_path(spec: ModelSpec) -> str:
    return os.path.join(CHECKPOINT_ROOT, spec.checkpoint_filename)


def _download_checkpoint(spec: ModelSpec) -> str:
    _ensure_directory(CHECKPOINT_ROOT)
    target = _checkpoint_path(spec)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target

    temp_path = f"{target}.part"
    response = requests.get(spec.checkpoint_url, stream=True, timeout=(10, 600))
    response.raise_for_status()
    with open(temp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
        raise RuntimeError(
            f"Downloaded checkpoint for '{spec.display_name}' is empty: {spec.checkpoint_url}"
        )

    os.replace(temp_path, target)
    return target


def _clone_model_config(config: dict) -> dict:
    return copy.deepcopy(config)


def _checkpoint_meta_config(spec: ModelSpec, checkpoint_path: str) -> dict:
    try:
        from mmcv import Config
    except ImportError as exc:
        raise RuntimeError(
            "EasyPortrait dependency error: module 'mmcv' is not installed. "
            "Install a mmsegmentation 0.30.0 compatible mmcv package."
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    config_text = meta.get("config") or meta.get("cfg")
    if not isinstance(config_text, str) or not config_text.strip():
        raise RuntimeError(
            f"No model config metadata was found inside '{spec.checkpoint_filename}'. "
            f"Upstream no longer publishes a source config for '{spec.display_name}'."
        )

    config = Config.fromstring(config_text, file_format=".py")
    model_dict = config.get("model")
    if not isinstance(model_dict, dict):
        raise RuntimeError(
            f"Checkpoint metadata for '{spec.display_name}' does not contain a usable model config."
        )

    model_dict = copy.deepcopy(model_dict)
    if "pretrained" in model_dict:
        model_dict["pretrained"] = None
    backbone = model_dict.get("backbone")
    if isinstance(backbone, dict) and "init_cfg" in backbone:
        backbone["init_cfg"] = None

    return {
        "model": model_dict,
        "test_pipeline": _test_pipeline(spec.input_size),
        "cudnn_benchmark": True,
    }


class EasyPortraitModelCache:
    _models: dict[tuple[str, tuple[str, ...]], object] = {}
    _lock = threading.Lock()

    @classmethod
    def load(cls, spec: ModelSpec):
        providers = _onnx_providers()
        cache_key = (spec.key, tuple(providers))
        with cls._lock:
            if cache_key in cls._models:
                return cls._models[cache_key]

            model = _init_model(spec, providers)
            cls._models[cache_key] = model
            return model


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "EasyPortrait dependency error: module 'onnxruntime' is not installed. "
            "Install this custom node's requirements before using it."
        ) from exc

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _onnx_filename(spec: ModelSpec) -> str:
    return f"{spec.key}.onnx"


def _onnx_path(spec: ModelSpec) -> str:
    return os.path.join(ONNX_ROOT, _onnx_filename(spec))


def _hf_headers() -> dict[str, str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def _download_onnx(spec: ModelSpec) -> str:
    _ensure_directory(ONNX_ROOT)
    target = _onnx_path(spec)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target

    url = f"{ONNX_REPO_URL}/{_onnx_filename(spec)}"
    temp_path = f"{target}.part"
    response = requests.get(
        url,
        headers=_hf_headers(),
        stream=True,
        timeout=(10, 600),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Could not download ONNX model '{_onnx_filename(spec)}' from "
            f"{ONNX_REPO_ID}. If the repository is private, set HF_TOKEN before "
            "starting ComfyUI."
        ) from exc

    with open(temp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
        raise RuntimeError(f"Downloaded ONNX model is empty: {url}")

    os.replace(temp_path, target)
    return target


def _init_model(spec: ModelSpec, providers: list[str]):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "EasyPortrait dependency error: module 'onnxruntime' is not installed. "
            "Install this custom node's requirements before using it."
        ) from exc

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        _download_onnx(spec),
        sess_options=session_options,
        providers=providers,
    )
    return session


def _to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def _resize_segmentation(segmentation: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if segmentation.shape == (height, width):
        return segmentation
    resized = Image.fromarray(segmentation.astype(np.uint8)).resize(
        (width, height),
        Image.NEAREST,
    )
    return np.asarray(resized)


def _preprocess_onnx(image: np.ndarray, spec: ModelSpec) -> np.ndarray:
    image = _resize_image(image, spec.input_size).astype(np.float32)
    image = image[..., ::-1]
    image = (image - np.asarray(IMG_MEAN, dtype=np.float32)) / np.asarray(
        IMG_STD, dtype=np.float32
    )
    image = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    return image


def _segment_onnx(model, image: np.ndarray, spec: ModelSpec) -> np.ndarray:
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: _preprocess_onnx(image, spec)})[0]
    output = np.asarray(output)
    if output.ndim == 4:
        segmentation = output.argmax(axis=1)[0].astype(np.uint8)
    elif output.ndim == 3:
        segmentation = output[0].astype(np.uint8)
    elif output.ndim == 2:
        segmentation = output.astype(np.uint8)
    else:
        raise RuntimeError(f"Unexpected EasyPortrait ONNX output shape: {output.shape}")
    return _resize_segmentation(segmentation, image.shape[:2])


def _to_mask(segmentation: np.ndarray, spec: ModelSpec) -> np.ndarray:
    if spec.task == TASK_PORTRAIT:
        return (segmentation == 1).astype(np.float32)
    return (segmentation > 0).astype(np.float32)


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def _label_map(spec: ModelSpec) -> dict[str, int]:
    return {_normalize_label(label): index for index, label in enumerate(CLASS_NAMES[spec.task])}


def _parse_labels(
    labels: str,
    spec: ModelSpec,
    *,
    person: bool,
    skin: bool,
    left_brow: bool,
    right_brow: bool,
    left_eye: bool,
    right_eye: bool,
    lips: bool,
    teeth: bool,
    background: bool,
) -> list[tuple[str, int]]:
    label_map = _label_map(spec)
    if labels and labels.strip():
        raw_labels = labels.split(",")
    else:
        raw_labels = [
            label
            for label, enabled in (
                ("person", person),
                ("skin", skin),
                ("left brow", left_brow),
                ("right brow", right_brow),
                ("left eye", left_eye),
                ("right eye", right_eye),
                ("lips", lips),
                ("teeth", teeth),
                ("background", background),
            )
            if enabled
        ]

    selected = []
    seen = set()
    for raw_label in raw_labels:
        normalized = _normalize_label(str(raw_label))
        if not normalized or normalized in seen:
            continue
        if normalized not in label_map:
            continue
        selected.append((CLASS_NAMES[spec.task][label_map[normalized]], label_map[normalized]))
        seen.add(normalized)

    if selected:
        return selected

    fallback = "person" if spec.task == TASK_PORTRAIT else "skin"
    return [(fallback, label_map[fallback])]


def _mask_for_label(segmentation: np.ndarray, label_index: int) -> np.ndarray:
    return (segmentation == label_index).astype(np.float32)


def _make_binary_mask(segmentation: np.ndarray, labels: list[tuple[str, int]]) -> np.ndarray:
    selected_indices = [label_index for _, label_index in labels]
    return np.isin(segmentation, selected_indices).astype(np.float32)


def _make_preview(
    image: np.ndarray,
    segmentation: np.ndarray,
    labels: list[tuple[str, int]],
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    for label, label_index in labels:
        mask = segmentation == label_index
        if not np.any(mask):
            continue
        color = LABEL_COLORS.get(label, np.array([255, 140, 0], dtype=np.uint8)).astype(
            np.float32
        )
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
    return np.clip(overlay, 0, 255).astype(np.uint8)


class EasyPortraitSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (MODEL_CHOICES, {"default": MODEL_CHOICES[0]}),
                "mode": (MASK_MODES, {"default": "binary"}),
                "person": ("BOOLEAN", {"default": True}),
                "skin": ("BOOLEAN", {"default": False}),
                "left_brow": ("BOOLEAN", {"default": False}),
                "right_brow": ("BOOLEAN", {"default": False}),
                "left_eye": ("BOOLEAN", {"default": False}),
                "right_eye": ("BOOLEAN", {"default": False}),
                "lips": ("BOOLEAN", {"default": False}),
                "teeth": ("BOOLEAN", {"default": False}),
                "background": ("BOOLEAN", {"default": False}),
                "labels": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional comma-separated labels",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "segment"
    CATEGORY = "EasyPortrait"

    def segment(
        self,
        image,
        model_name,
        mode,
        person,
        skin,
        left_brow,
        right_brow,
        left_eye,
        right_eye,
        lips,
        teeth,
        background,
        labels,
    ):
        spec = MODELS_BY_NAME[model_name]
        model = EasyPortraitModelCache.load(spec)
        selected_labels = _parse_labels(
            labels,
            spec,
            person=person,
            skin=skin,
            left_brow=left_brow,
            right_brow=right_brow,
            left_eye=left_eye,
            right_eye=right_eye,
            lips=lips,
            teeth=teeth,
            background=background,
        )

        masks = []
        previews = []

        for image_item in image:
            image_np = _to_numpy_image(image_item)
            segmentation = _segment_onnx(model, image_np, spec)

            if mode == "layers":
                for _, label_index in selected_labels:
                    masks.append(torch.from_numpy(_mask_for_label(segmentation, label_index)))
            else:
                masks.append(torch.from_numpy(_make_binary_mask(segmentation, selected_labels)))

            preview = _make_preview(image_np, segmentation, selected_labels)
            previews.append(torch.from_numpy(preview.astype(np.float32) / 255.0))

        return (torch.stack(masks, dim=0), torch.stack(previews, dim=0))


NODE_CLASS_MAPPINGS = {
    "EasyPortraitSegment": EasyPortraitSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyPortraitSegment": "EasyPortrait Segment",
}
