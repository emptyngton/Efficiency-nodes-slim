# Efficiency Nodes Slim (Minimal Set)

Slim refactor focused on a fast, minimal set of nodes:

- Efficient Loader
- KSampler (Efficient)
- LoRA Stacker
- Control Net Stacker
- HighRes-Fix Script

Minimum ComfyUI version: v0.3.50.

Notes:
- Optional: `comfyui_controlnet_aux` enables the HighRes-Fix preprocessor dropdown.
- HighRes-Fix `control_net_name` accepts model names or full paths; set Use ControlNet to true to reveal related options.
- Efficient Loader uses baked VAE automatically when selected.

Install:
- Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

