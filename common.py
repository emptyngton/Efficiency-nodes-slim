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