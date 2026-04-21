# ComfyUI EasyPortrait

ComfyUI custom node for portrait segmentation and face parsing using ONNX exports of the pretrained EasyPortrait checkpoints.

The original models were published by the EasyPortrait authors in [hukenovs/easyportrait](https://github.com/hukenovs/easyportrait). This extension provides a lightweight ONNX Runtime path so users do not need to install the old `mmsegmentation` / `mmcv-full` stack.

## Features

- Portrait segmentation and face parsing in one node.
- Automatic ONNX model download from [sadzip/EasyPortrait-ONNX](https://huggingface.co/sadzip/EasyPortrait-ONNX).
- Lightweight runtime dependencies: `onnxruntime`, `Pillow`, and `requests`.
- Label filtering with checkbox inputs or comma-separated text.
- Two mask modes:
  - `binary`: selected labels are merged into one mask per input image.
  - `layers`: each selected label is returned as a separate mask in the output batch.
- Preview overlay with a distinct color per selected label.

## Installation

Clone this repository into `ComfyUI/custom_nodes` and install the requirements into the same Python environment used by ComfyUI:

```bash
/venv/main/bin/python -m pip install -r custom_nodes/ComfyUI-EasyPortrait/requirements.txt
```

Restart ComfyUI after installation.

## Models

ONNX files are downloaded on first use into:

```text
ComfyUI/models/easyportrait/onnx
```

If the Hugging Face repository is private, start ComfyUI with `HF_TOKEN` set in the environment.

The ONNX set contains the reproducible EasyPortrait checkpoints with available model configs. Three upstream README-only checkpoints are not exposed by default because they do not include enough architecture metadata for reliable conversion: `extremec3net_ps`, `sinet_ps`, and `ehanet_fp`.

## Inputs

- `image`: ComfyUI image batch.
- `model_name`: ONNX model to run.
- `mode`: `binary` or `layers`.
- `person`, `skin`, `left_brow`, `right_brow`, `left_eye`, `right_eye`, `lips`, `teeth`, `background`: checkbox label selectors.
- `labels`: optional comma-separated labels. When this field is non-empty, it overrides the checkbox selectors.

Available labels:

- Portrait models: `background`, `person`
- Face parsing models: `background`, `skin`, `left brow`, `right brow`, `left eye`, `right eye`, `lips`, `teeth`

Examples for `labels`:

```text
person
skin,lips,teeth
left eye, right eye
```

## Outputs

- `mask`: ComfyUI `MASK` batch.
  - In `binary` mode: one combined mask per input image.
  - In `layers` mode: one mask per selected label per input image.
- `preview`: image preview batch with selected labels overlaid in label-specific colors.

## Development

The runtime node does not require `mmsegmentation` or `mmcv-full`. They are only needed to regenerate ONNX files from the original checkpoints.

The development converter is:

```bash
/venv/main/bin/python custom_nodes/ComfyUI-EasyPortrait/scripts/export_onnx.py --device cpu --overwrite
```

It exports ONNX files, validates ONNX Runtime outputs against PyTorch/mmseg outputs, and can upload the validated files to Hugging Face with `--upload`.

## Credits

Thanks to the EasyPortrait authors for training and releasing the original checkpoints and code:

- Original repository: [hukenovs/easyportrait](https://github.com/hukenovs/easyportrait)

Co-authors of this ONNX/ComfyUI integration:

- preposition17
- OpenAI Codex
