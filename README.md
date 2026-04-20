# ComfyUI EasyPortrait

Custom node for portrait segmentation and face parsing using the pretrained checkpoints published in `hukenovs/easyportrait`.

Inputs:
- `image`
- `model_name`

Outputs:
- `mask`
- `preview`

Notes:
- Checkpoints are downloaded automatically into `models/easyportrait/checkpoints`.
- The node uses `mmsegmentation 0.30.0` style inference.
- `mmsegmentation` alone is not enough; a compatible `mmcv` package must also be installed.
- Upstream `main` currently publishes source configs for 14 models directly. The remaining README-only checkpoints are loaded via checkpoint metadata when possible.
