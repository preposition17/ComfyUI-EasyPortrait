from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
COMFYUI_DIR = REPO_DIR.parents[1]
if str(COMFYUI_DIR) not in sys.path:
    sys.path.insert(0, str(COMFYUI_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from easy_portrait import (  # noqa: E402
    CHECKPOINT_ROOT,
    CONFIG_BUILDERS,
    IMG_MEAN,
    IMG_STD,
    MODEL_SPECS,
    ONNX_MODEL_SPECS,
    ONNX_REPO_ID,
    ONNX_ROOT,
    _checkpoint_meta_config,
    _clone_model_config,
    _download_checkpoint,
    _replace_sync_batchnorm,
)


class EasyPortraitOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_decode(image, [{}])


def _specs_by_key():
    return {spec.key: spec for spec in MODEL_SPECS}


def _build_mmseg_model(spec, device: str):
    from mmcv import Config
    from mmseg.apis import init_segmentor

    checkpoint_path = _download_checkpoint(spec)
    try:
        config_dict = _checkpoint_meta_config(spec, checkpoint_path)
    except RuntimeError:
        if spec.builder_name is None:
            raise
        config_dict = CONFIG_BUILDERS[spec.builder_name](spec)
    if isinstance(config_dict.get("model"), dict) and "test_cfg" in config_dict["model"]:
        config_dict.pop("test_cfg", None)

    config = Config(_clone_model_config(config_dict))
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        model = init_segmentor(config, checkpoint=checkpoint_path, device=device)
    finally:
        torch.load = original_torch_load
    model = _replace_sync_batchnorm(model)
    model.eval()
    return model


def _export_one(spec, output_dir: Path, device: str, opset: int, overwrite: bool):
    output_path = output_dir / f"{spec.key}.onnx"
    if output_path.exists() and not overwrite:
        return output_path

    model = _build_mmseg_model(spec, device)
    wrapper = EasyPortraitOnnxWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, spec.input_size, spec.input_size, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={
                "image": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
    return output_path


def _validate_one(spec, onnx_path: Path, device: str, atol: float):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_path))

    model = _build_mmseg_model(spec, device)
    wrapper = EasyPortraitOnnxWrapper(model).to(device).eval()
    rng = np.random.default_rng(20260421)
    image = rng.normal(
        0.0,
        1.0,
        size=(1, 3, spec.input_size, spec.input_size),
    ).astype(np.float32)

    with torch.no_grad():
        torch_output = wrapper(torch.from_numpy(image).to(device)).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_output = session.run([output_name], {input_name: image})[0]

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    pred_match = bool(
        np.array_equal(torch_output.argmax(axis=1), onnx_output.argmax(axis=1))
    )
    if max_abs_diff > atol and not pred_match:
        raise RuntimeError(
            f"{spec.key} validation failed: max_abs_diff={max_abs_diff:.6g}, "
            f"pred_match={pred_match}"
        )

    return {
        "key": spec.key,
        "file": onnx_path.name,
        "display_name": spec.display_name,
        "task": spec.task,
        "checkpoint": spec.checkpoint_filename,
        "classes": int(torch_output.shape[1]),
        "max_abs_diff": max_abs_diff,
        "argmax_match": pred_match,
    }


def _write_manifest(results, output_dir: Path, opset: int):
    manifest = {
        "format": "onnx",
        "opset": opset,
        "input": {
            "name": "image",
            "shape": ["batch", 3, "model_size", "model_size"],
            "dtype": "float32",
            "preprocessing": {
                "source_image": "uint8 RGB",
                "channel_order": "RGB input reversed to BGR before normalization",
                "mean": IMG_MEAN,
                "std": IMG_STD,
            },
        },
        "output": {
            "name": "logits",
            "shape": ["batch", "classes", "model_size", "model_size"],
            "dtype": "float32",
        },
        "source_checkpoints_dir": CHECKPOINT_ROOT,
        "models": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _upload(output_dir: Path, repo_id: str, token: str | None):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Upload EasyPortrait ONNX models",
    )


def main():
    parser = argparse.ArgumentParser(description="Export EasyPortrait checkpoints to ONNX.")
    parser.add_argument("--output-dir", default=ONNX_ROOT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--only", nargs="*", default=None, help="Model keys to export.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Also try models without a known mmseg config. These usually fail.",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=ONNX_REPO_ID)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    specs = _specs_by_key()
    default_specs = MODEL_SPECS if args.include_unsupported else ONNX_MODEL_SPECS
    selected = [specs[key] for key in args.only] if args.only else default_specs

    results = []
    for spec in selected:
        print(f"[EasyPortrait] {spec.key}: export")
        onnx_path = output_dir / f"{spec.key}.onnx"
        if not args.validate_only:
            onnx_path = _export_one(
                spec,
                output_dir=output_dir,
                device=args.device,
                opset=args.opset,
                overwrite=args.overwrite,
            )

        print(f"[EasyPortrait] {spec.key}: validate")
        results.append(_validate_one(spec, onnx_path, device=args.device, atol=args.atol))

    manifest_path = _write_manifest(results, output_dir, args.opset)
    print(f"[EasyPortrait] wrote {manifest_path}")

    if args.upload:
        token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        _upload(output_dir, args.repo_id, token)
        print(f"[EasyPortrait] uploaded {output_dir} to {args.repo_id}")


if __name__ == "__main__":
    main()
