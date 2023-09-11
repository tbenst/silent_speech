##
%pip install --quiet --upgrade diffusers transformers accelerate invisible_watermark mediapy
##
use_refiner = True
import mediapy as media
import random, os
import sys
import torch

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    )

if use_refiner:
  refiner = DiffusionPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-refiner-1.0",
      text_encoder_2=pipe.text_encoder_2,
      vae=pipe.vae,
      torch_dtype=torch.float16,
      use_safetensors=True,
      variant="fp16",
  )

  refiner = refiner.to("cuda")

  pipe.enable_model_cpu_offload()
else:
  pipe = pipe.to("cuda")
##
def promp_to_fn(prompt, save_dir, seed, use_refiner=use_refiner):
    fn = f"{save_dir}/{prompt.replace(' ', '_')}_{seed}"
    if use_refiner:
        fn += "_refined"
    fn += ".png"
    return fn

def fn_to_prompt(fn):
    "Return prompt, save_dir, seed from filename"
    name = fn.split("/")[-1]
    use_refiner = "_refined" in name
    name = name.replace("_refined", "")
    prompt, seed = name.rsplit("_", 1)
    prompt = prompt.replace("_", " ")
    seed = int(seed.rsplit(".")[0])
    save_dir = os.path.dirname(fn)
    return prompt, save_dir, seed
##
save_dir = "/data/Dropbox/Photos/stable_diffusion/2023-09-05_green-light"
# prompt = "Picture of a book morphing into an audio waveform"
prompt = "Painting of a book morphing into an audio waveform"
seed = random.randint(0, sys.maxsize)

images = pipe(
    prompt = prompt,
    output_type = "latent" if use_refiner else "pil",
    generator = torch.Generator("cuda").manual_seed(seed),
    ).images

if use_refiner:
  images = refiner(
      prompt = prompt,
      image = images,
      ).images

print(f"prompt: {prompt}; seed: {seed}")
media.show_images(images)
fn = promp_to_fn(prompt, save_dir, seed)
##
images[0].save(fn)
##
