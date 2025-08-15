# Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
# by Luciano Cirino (Discord: TSC#9184) - April 2023 - October 2023
# https://github.com/LucianoCirino/efficiency-nodes-comfyui

from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import ast
from pathlib import Path
from importlib import import_module
import os
import sys
import copy
import subprocess
import json
import psutil

from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
from nodes import LatentUpscaleBy, KSampler, KSamplerAdvanced, VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled, \
    ImageScaleBy, CLIPSetLastLayer, CLIPTextEncode, ControlNetLoader, ControlNetApply, ControlNetApplyAdvanced, \
    PreviewImage, MAX_RESOLUTION
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats
sys.path.remove(comfy_dir)

# Append my_dir to sys.path & import files
sys.path.append(my_dir)
from tsc_utils import *
from .py import smZ_cfg_denoiser
from .py import smZ_rng_source
from .py import cg_mixed_seed_noise
from .py import city96_latent_upscaler
from .py import ttl_nn_latent_upscaler
from .py import bnk_tiled_samplers
from .py import bnk_adv_encode
sys.path.remove(my_dir)

from comfy import samplers
# Append custom_nodes_dir to sys.path
sys.path.append(custom_nodes_dir)

# GLOBALS
REFINER_CFG_OFFSET = 0 #Refiner CFG Offset

# Monkey patch schedulers
SCHEDULER_NAMES = samplers.SCHEDULER_NAMES + ["AYS SD1", "AYS SDXL", "AYS SVD"]
SCHEDULERS = samplers.KSampler.SCHEDULERS + ["AYS SD1", "AYS SDXL", "AYS SVD"]

########################################################################################################################
# Common function for encoding prompts
def encode_prompts(positive_prompt, negative_prompt, token_normalization, weight_interpretation, clip, clip_skip,
                   refiner_clip, refiner_clip_skip, ascore, is_sdxl, empty_latent_width, empty_latent_height,
                   return_type="both"):

    positive_encoded = negative_encoded = refiner_positive_encoded = refiner_negative_encoded = None

    # Process base encodings if needed
    if return_type in ["base", "both"]:
        clip = CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]

        positive_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(clip, positive_prompt, token_normalization, weight_interpretation)[0]
        negative_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(clip, negative_prompt, token_normalization, weight_interpretation)[0]

    # Process refiner encodings if needed
    if return_type in ["refiner", "both"] and is_sdxl and refiner_clip and refiner_clip_skip and ascore:
        refiner_clip = CLIPSetLastLayer().set_last_layer(refiner_clip, refiner_clip_skip)[0]

        refiner_positive_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(refiner_clip, positive_prompt, token_normalization, weight_interpretation)[0]
        refiner_positive_encoded = bnk_adv_encode.AddCLIPSDXLRParams().encode(refiner_positive_encoded, empty_latent_width, empty_latent_height, ascore[0])[0]

        refiner_negative_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(refiner_clip, negative_prompt, token_normalization, weight_interpretation)[0]
        refiner_negative_encoded = bnk_adv_encode.AddCLIPSDXLRParams().encode(refiner_negative_encoded, empty_latent_width, empty_latent_height, ascore[1])[0]

    # Return results based on return_type
    if return_type == "base":
        return positive_encoded, negative_encoded, clip
    elif return_type == "refiner":
        return refiner_positive_encoded, refiner_negative_encoded, refiner_clip
    elif return_type == "both":
        return positive_encoded, negative_encoded, clip, refiner_positive_encoded, refiner_negative_encoded, refiner_clip

########################################################################################################################
def preprocess_prompt(prompt, comment_marker="##"):
    """
    Preprocess the prompt by removing any tokens that start with the specified comment marker.
    Tokens are assumed to be separated by commas.
    """
    if not prompt:
        return prompt
    # Split the prompt into individual tokens
    tags = [tag.strip() for tag in prompt.split(',')]
    # Filter out tokens that start with the comment marker
    active_tags = [tag for tag in tags if not tag.startswith(comment_marker)]
    # Reassemble the prompt
    return ', '.join(active_tags)

########################################################################################################################
class TSC_EfficientLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # Concise tag fields:
                "essential_tags": ("STRING", {"default": "", "multiline": True}),
                "camera_character": ("STRING", {"default": "", "multiline": True}),
                "position_powertags": ("STRING", {"default": "", "multiline": True}),
                "clothing": ("STRING", {"default": "", "multiline": True}),
                "environment_lighting": ("STRING", {"default": "", "multiline": True}),
                "quality_tags": ("STRING", {"default": "", "multiline": True}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 262144}),
                "negative": ("STRING", {"default": "CLIP_NEGATIVE", "multiline": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", ),
                "cnet_stack": ("CONTROL_NET_STACK",)
            },
            "hidden": {
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "DEPENDENCIES",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "CLIP", "DEPENDENCIES", )
    FUNCTION = "efficientloader"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        essential_tags, camera_character, position_powertags, clothing, environment_lighting,
                        quality_tags, negative, token_normalization, weight_interpretation, empty_latent_width,
                        empty_latent_height, batch_size, lora_stack=None, cnet_stack=None, refiner_name="None",
                        ascore=None, prompt=None, my_unique_id=None, loader_type="regular"):

        # Helper function: if a field does not end with a comma, add one.
        def ensure_comma(field):
            f = field.strip()
            if f and not f.endswith(','):
                return f + ","
            return f

        # Preprocess prompt fields to remove commented out tags (using '##' as the comment marker)
        essential_tags = preprocess_prompt(essential_tags, comment_marker="##")
        camera_character = preprocess_prompt(camera_character, comment_marker="##")
        position_powertags = preprocess_prompt(position_powertags, comment_marker="##")
        clothing = preprocess_prompt(clothing, comment_marker="##")
        environment_lighting = preprocess_prompt(environment_lighting, comment_marker="##")
        quality_tags = preprocess_prompt(quality_tags, comment_marker="##")

        # Process and concatenate each field.
        fields = [
            ensure_comma(essential_tags),
            ensure_comma(camera_character),
            ensure_comma(position_powertags),
            ensure_comma(clothing),
            ensure_comma(environment_lighting),
            ensure_comma(quality_tags)
        ]
        positive = " ".join(fields).strip()
        # Remove the trailing comma from the final string, if present.
        if positive.endswith(','):
            positive = positive[:-1]

        # Clean globally stored objects
        globals_cleanup(prompt)

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()

        # Retrieve cache numbers
        vae_cache, ckpt_cache, lora_cache, refn_cache = get_cache_numbers("Efficient Loader")

        vae = None
        if lora_name != "None" or lora_stack:
            lora_params = []
            if lora_name != "None":
                lora_params.append((lora_name, lora_model_strength, lora_clip_strength))
            if lora_stack:
                lora_params.extend(lora_stack)
            model, clip = load_lora(lora_params, ckpt_name, my_unique_id, cache=lora_cache,
                                      ckpt_cache=ckpt_cache, cache_overwrite=True)
            if vae_name == "Baked VAE":
                # Prefer baked VAE from the loaded checkpoint; fall back to loading if missing
                vae = get_bvae_by_ckpt_name(ckpt_name)
                if vae is None:
                    _, _, baked_vae = load_checkpoint(ckpt_name, my_unique_id, cache=ckpt_cache, cache_overwrite=True)
                    vae = baked_vae
        else:
            model, clip, vae = load_checkpoint(ckpt_name, my_unique_id, cache=ckpt_cache, cache_overwrite=True)
            lora_params = None

        if refiner_name != "None":
            refiner_model, refiner_clip, _ = load_checkpoint(refiner_name, my_unique_id, output_vae=False,
                                                             cache=refn_cache, cache_overwrite=True, ckpt_type="refn")
        else:
            refiner_model = refiner_clip = None

        refiner_clip_skip = clip_skip[1] if loader_type == "sdxl" else None
        clip_skip = clip_skip[0] if loader_type == "sdxl" else clip_skip

        # Encode prompt using the concatenated positive string.
        positive_encoded, negative_encoded, clip, refiner_positive_encoded, refiner_negative_encoded, refiner_clip = \
            encode_prompts(positive, negative, token_normalization, weight_interpretation, clip, clip_skip,
                           refiner_clip, refiner_clip_skip, ascore, loader_type == "sdxl",
                           empty_latent_width, empty_latent_height)

        if cnet_stack:
            controlnet_conditioning = TSC_Apply_ControlNet_Stack().apply_cnet_stack(positive_encoded, negative_encoded, cnet_stack)
            positive_encoded, negative_encoded = controlnet_conditioning[0], controlnet_conditioning[1]

        if vae_name != "Baked VAE":
            vae = load_vae(vae_name, my_unique_id, cache=vae_cache, cache_overwrite=True)
        else:
            # Ensure we have a VAE object if the checkpoint exposes one baked-in
            if vae is None:
                _, _, baked_vae = load_checkpoint(ckpt_name, my_unique_id, cache=ckpt_cache, cache_overwrite=True)
                vae = baked_vae

        dependencies = (vae_name, ckpt_name, clip, clip_skip, refiner_name, refiner_clip, refiner_clip_skip,
                        positive, negative, token_normalization, weight_interpretation, ascore,
                        empty_latent_width, empty_latent_height, lora_params, cnet_stack)

        print_loaded_objects_entries(my_unique_id, prompt)

        if loader_type == "regular":
            return (model, positive_encoded, negative_encoded, {"samples": latent}, vae, clip, dependencies)
        elif loader_type == "sdxl":
            return ((model, clip, positive_encoded, negative_encoded, refiner_model, refiner_clip,
                     refiner_positive_encoded, refiner_negative_encoded), {"samples": latent}, vae, dependencies)