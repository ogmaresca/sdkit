from contextlib import nullcontext

import torch
from pytorch_lightning import seed_everything
from tqdm import trange

from sdkit import Context
from sdkit.utils import (
    apply_color_profile,
    base64_str_to_img,
    gc,
    get_image_latent_and_mask,
    latent_samples_to_images,
    resize_img,
)

from .prompt_parser import get_cond_and_uncond
from .sampler import make_samples

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def generate_images(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    init_image=None,
    init_image_mask=None,
    prompt_strength: float = 0.8,
    preserve_init_image_color_profile=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    hypernetwork_strength: float = 0,
    lora_alpha: float = 0,
    sampler_params={},
    callback=None,
):
    req_args = locals()

    try:
        images = []

        seed_everything(seed)
        precision_scope = torch.autocast if context.half_precision else nullcontext

        if "stable-diffusion" not in context.models:
            raise RuntimeError(
                "The model for Stable Diffusion has not been loaded yet! If you've tried to load it, please check the logs above this message for errors (while loading the model)."
            )

        model = context.models["stable-diffusion"]

        if context.test_diffusers:
            return make_with_diffusers(
                context,
                prompt,
                negative_prompt,
                seed,
                width,
                height,
                num_outputs,
                num_inference_steps,
                guidance_scale,
                init_image,
                init_image_mask,
                prompt_strength,
                # preserve_init_image_color_profile,
                sampler_name,
                # hypernetwork_strength,
                lora_alpha,
                # sampler_params,
                callback,
            )

        if "hypernetwork" in context.models:
            context.models["hypernetwork"]["hypernetwork_strength"] = hypernetwork_strength

        with precision_scope("cuda"):
            cond, uncond = get_cond_and_uncond(prompt, negative_prompt, num_outputs, model)

        generate_fn = txt2img if init_image is None else img2img
        common_sampler_params = {
            "context": context,
            "sampler_name": sampler_name,
            "seed": seed,
            "batch_size": num_outputs,
            "shape": [4, height // 8, width // 8],
            "cond": cond,
            "uncond": uncond,
            "guidance_scale": guidance_scale,
            "sampler_params": sampler_params,
            "callback": callback,
        }

        with torch.no_grad(), precision_scope("cuda"):
            for _ in trange(1, desc="Sampling"):
                images += generate_fn(common_sampler_params.copy(), **req_args)
                gc(context)

        return images
    finally:
        context.init_image_latent, context.init_image_mask_tensor = None, None


def txt2img(params: dict, context: Context, num_inference_steps, **kwargs):
    params.update(
        {
            "steps": num_inference_steps,
        }
    )

    samples = make_samples(**params)
    return latent_samples_to_images(context, samples)


def img2img(
    params: dict,
    context: Context,
    num_inference_steps,
    num_outputs,
    width,
    height,
    init_image,
    init_image_mask,
    prompt_strength,
    preserve_init_image_color_profile,
    **kwargs,
):
    init_image = get_image(init_image)
    init_image_mask = get_image(init_image_mask)

    if not hasattr(context, "init_image_latent") or context.init_image_latent is None:
        context.init_image_latent, context.init_image_mask_tensor = get_image_latent_and_mask(
            context, init_image, init_image_mask, width, height, num_outputs
        )

    params.update(
        {
            "steps": num_inference_steps,
            "init_image_latent": context.init_image_latent,
            "mask": context.init_image_mask_tensor,
            "prompt_strength": prompt_strength,
        }
    )

    samples = make_samples(**params)
    images = latent_samples_to_images(context, samples)

    if preserve_init_image_color_profile:
        for i, img in enumerate(images):
            images[i] = apply_color_profile(init_image, img)

    return images


def make_with_diffusers(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    init_image=None,
    init_image_mask=None,
    prompt_strength: float = 0.8,
    # preserve_init_image_color_profile=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    # hypernetwork_strength: float = 0,
    lora_alpha: float = 0,
    # sampler_params={},
    callback=None,
):
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
    )

    from sdkit.generate.sampler import diffusers_samplers
    from sdkit.models.model_loader.lora import apply_lora_model
    from sdkit.utils import log

    model = context.models["stable-diffusion"]
    if context.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator(context.device).manual_seed(seed)

    cmd = {
        "guidance_scale": guidance_scale,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_outputs,
    }
    if init_image:
        cmd["image"] = get_image(init_image).convert("RGB")
        cmd["image"] = resize_img(cmd["image"], width, height, clamp_to_64=True)
        cmd["strength"] = prompt_strength
    if init_image_mask:
        cmd["mask_image"] = get_image(init_image_mask).convert("RGB")
        cmd["mask_image"] = resize_img(cmd["mask_image"], width, height, clamp_to_64=True)

    if init_image:
        operation_to_apply = "inpainting" if init_image_mask else "img2img"
    else:
        operation_to_apply = "txt2img"

    if operation_to_apply not in model:
        if "inpainting" in model and len(model) == 1:
            raise RuntimeError(
                f"This model does not support {operation_to_apply}! This model requires an initial image and mask."
            )

        raise NotImplementedError(
            f"This model does not support {operation_to_apply}! Supported operations: {model.keys()}"
        )

    operation_to_apply = model[operation_to_apply]
    if diffusers_samplers.samplers.get(sampler_name) is None:
        raise NotImplementedError(f"The sampler '{sampler_name}' is not supported (yet)!")

    operation_to_apply.scheduler = diffusers_samplers.samplers[sampler_name]
    log.info(f"Using sampler: {operation_to_apply.scheduler} because of {sampler_name}")

    if isinstance(operation_to_apply, StableDiffusionInpaintPipelineLegacy) or isinstance(
        operation_to_apply, StableDiffusionImg2ImgPipeline
    ):
        del cmd["width"]
        del cmd["height"]
    elif isinstance(operation_to_apply, StableDiffusionInpaintPipeline):
        del cmd["strength"]

    cmd["callback"] = lambda i, t, x_samples: callback(x_samples, i, operation_to_apply) if callback else None

    # apply the LoRA (if necessary)
    if context.models.get("lora"):
        log.info("Applying LoRA..")
        if hasattr(context, "_last_lora_alpha"):
            apply_lora_model(context, -context._last_lora_alpha)  # undo the last LoRA apply

        apply_lora_model(context, lora_alpha)
        context._last_lora_alpha = lora_alpha

    log.info("Parsing the prompt..")

    # make the prompt embeds
    compel = model["compel"]

    # temporary hack until compel 1.1.4 is released
    if hasattr(operation_to_apply.text_encoder, "_hf_hook"):
        [m._hf_hook.pre_forward(m) for m in operation_to_apply.text_encoder.modules() if hasattr(m, "_hf_hook")]
        print(compel.device)

    log.info("compel is ready")
    cmd["prompt_embeds"] = compel(prompt)

    if hasattr(operation_to_apply.text_encoder, "_hf_hook"):
        [m._hf_hook.pre_forward(m) for m in operation_to_apply.text_encoder.modules() if hasattr(m, "_hf_hook")]
        print(compel.device)

    log.info("Made prompt embeds")
    cmd["negative_prompt_embeds"] = compel(negative_prompt)

    if hasattr(operation_to_apply.text_encoder, "_hf_hook"):
        [m._hf_hook.pre_forward(m) for m in operation_to_apply.text_encoder.modules() if hasattr(m, "_hf_hook")]
        print(compel.device)

    log.info("Made negative prompt embeds")
    cmd["prompt_embeds"], cmd["negative_prompt_embeds"] = compel.pad_conditioning_tensors_to_same_length(
        [cmd["prompt_embeds"], cmd["negative_prompt_embeds"]]
    )

    log.info("Done parsing the prompt")

    # apply
    log.info(f"applying: {operation_to_apply}")
    log.info(f"Running on diffusers: {cmd}")

    return operation_to_apply(**cmd).images


def get_image(img):
    if not isinstance(img, str):
        return img

    if img.startswith("data:image"):
        return base64_str_to_img(img)

    import os

    if os.path.exists(img):
        from PIL import Image

        return Image.open(img)
