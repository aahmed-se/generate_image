import argparse
import warnings
from datetime import datetime

import torch
import yaml
from huggingface_hub import hf_hub_download
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverSDEScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)

warnings.filterwarnings('ignore')


def _create_scheduler(sampler: str):
    scheduler = sampler
    if scheduler not in ['ddim',
                         'pndm',
                         'ddpm',
                         'lms',
                         'euler',
                         'euler_a',
                         'kdpm2',
                         'dpm++',
                         'dpm++_2s',
                         'dpm++_2m',
                         'dpm++_sde',
                         'dpm++_2m_sde',
                         'dpm++_2s_k',
                         'dpm++_2m_k',
                         'dpm++_sde_k',
                         'dpm++_2m_sde_k']:
        print(f"unsupported scheduler '{scheduler}', falling back to ddim")
        scheduler = 'ddim'

    if scheduler == 'ddim':
        return DDIMScheduler
    elif scheduler == 'dpm++_2s':
        return DPMSolverSinglestepScheduler.from_config({}, use_karras_sigmas=False)
    elif scheduler == 'dpm++_2s_k':
        return DPMSolverSinglestepScheduler.from_config({}, use_karras_sigmas=True)
    elif scheduler == 'dpm++' or scheduler == 'dpm++_2m':
        return DPMSolverMultistepScheduler.from_config({}, algorithm_type="dpmsolver++", use_karras_sigmas=False)
    elif scheduler == 'dpm++_2m_k':
        return DPMSolverMultistepScheduler.from_config({}, use_karras_sigmas=True)
    elif scheduler == 'dpm++_sde':
        return DPMSolverSDEScheduler.from_config({}, use_karras_sigmas=False, noise_sampler_seed=0)
    elif scheduler == 'dpm++_sde_k':
        return DPMSolverSDEScheduler.from_config({}, use_karras_sigmas=True, noise_sampler_seed=0)
    elif scheduler == 'dpm++_2m_sde':
        return DPMSolverMultistepScheduler.from_config({}, algorithm_type="sde-dpmsolver++", use_karras_sigmas=False)
    elif scheduler == 'dpm++_2m_sde_k':
        return DPMSolverMultistepScheduler.from_config({}, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
    elif scheduler == 'pndm':
        return PNDMScheduler
    elif scheduler == 'ddpm':
        return DDPMScheduler
    elif scheduler == 'lms':
        return LMSDiscreteScheduler
    elif scheduler == 'euler':
        return EulerDiscreteScheduler
    elif scheduler == 'euler_a':
        return EulerAncestralDiscreteScheduler
    elif scheduler == 'kdpm2':
        return KDPM2AncestralDiscreteScheduler
    else:
        raise ValueError(f"unknown scheduler '{scheduler}'")


def invoke(prompt: str,
           negative_prompt: str = "",
           steps: int = 20,
           scale: float = 5.0,
           seed: int = None,
           height: int = 768,
           width: int = 512,
           sampler: str = "dpm") -> str:
    model = hf_hub_download(repo_id="Remilistrasza/epiCRealism",
                            filename="epicrealism_naturalSinRC1VAE.safetensors"
                            )

    pipe = StableDiffusionPipeline.from_single_file(model,
                                                    torch_dtype=torch.float16,
                                                    use_safetensors=True,
                                                    scheduler=_create_scheduler(sampler))
    pipe = pipe.to("mps")
    pipe.safety_checker = None
    if seed is None:
        seed = torch.randint(0, 2 ** 32, (1,)).item()

    generator = torch.Generator("mps").manual_seed(seed)

    # Generate the image
    pipe_obj = pipe(prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance_scale=scale,
                    generator=generator)

    # Get the current timestamp
    timestamp = int(datetime.now().timestamp())

    # Include height, width, scale, and seed in the filename
    file_name = f"img_h{height}_w{width}_sc{scale}_se{seed}_{timestamp}.jpeg"

    # Save the image
    pipe_obj.images[0].save(file_name)

    # Return the filename
    return file_name


def read_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from textual prompts")
    parser.add_argument("-c", "--config", default="config.yaml", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config_path = args.config
    config = read_config(config_path)

    file_name = invoke(prompt=config["prompt"],
                       negative_prompt=config.get("negative_prompt", ""),
                       steps=config.get("steps", 20),
                       height=config.get("height", 768),
                       width=config.get("width", 512),
                       scale=config.get("scale", 5.0),
                       seed=config.get("seed", None),
                       sampler=config.get("sampler", "dpm"))

    print(file_name)
