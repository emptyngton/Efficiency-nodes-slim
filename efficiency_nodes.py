import os
import io

# Dynamically execute only the selected refactor chunks in this module's namespace

_here = os.path.dirname(os.path.abspath(__file__))
_chunks_dir = _here


def _exec_chunk(filename: str):
    path = os.path.join(_chunks_dir, filename)
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    # Ensure executed chunks see the package root as __file__ so their path math matches the original monolith
    exec_globals = globals()
    exec_globals['__file__'] = __file__
    exec(compile(code, path, 'exec'), exec_globals, exec_globals)


# Order matters: common imports/helpers → controlnet widgets → stackers → scripts → loader → ksampler
_exec_chunk('common.py')
_exec_chunk('controlnet_aux_options.py')
_exec_chunk(os.path.join('nodes', 'lora_stacker.py'))
_exec_chunk(os.path.join('nodes', 'control_net_stacker.py'))
_exec_chunk(os.path.join('nodes', 'apply_controlnet_stack.py'))
_exec_chunk(os.path.join('nodes', 'highres_fix.py'))
_exec_chunk(os.path.join('nodes', 'efficient_loader.py'))
_exec_chunk(os.path.join('nodes', 'ksampler_efficient.py'))


# Only expose the nodes you want
NODE_CLASS_MAPPINGS = {
    "KSampler (Efficient)": TSC_KSampler,
    "Efficient Loader": TSC_EfficientLoader,
    "LoRA Stacker": TSC_LoRA_Stacker,
    "Control Net Stacker": TSC_Control_Net_Stacker,
    "HighRes-Fix Script": TSC_HighRes_Fix,
}

# Optional display name mappings (keep empty or customize)
NODE_DISPLAY_NAME_MAPPINGS = {}


