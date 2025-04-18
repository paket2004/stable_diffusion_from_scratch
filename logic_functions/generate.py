sampler = "ddpm"
import torch
import sys
import os
import torch
# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logic_classes.ddpm_sampler import DDPMSampler
from .preload_model import preload_models_from_standard_weights
from tqdm import tqdm
from .time_embedding import get_time_embedding
from .rescale import rescale
WIDTH = 512
HEIGHT = 512
seed = 42
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
from transformers import CLIPTokenizer
DEVICE = "cpu"
merges_file = os.path.join("model_setup", "tokenizer_merges.txt")
tokenizer_name = os.path.join("model_setup", "tokenizer_vocab.json")
# print(tokenizer)

def generate(
    prompt,
    uncond_prompt="",
    strength=0.8,
    do_cfg=True,
    input_image=None,
    sampler_name=sampler,
    cfg_scale=7.5,
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    print("sth")
    sampler = "ddpm"
    model_file = os.path.join("model_setup", "v1-5-pruned-emaonly.ckpt")
    with torch.no_grad():
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        print("sth2")
        seed = 42
        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        print(seed)
        generator.manual_seed(seed)
        models = preload_models_from_standard_weights(model_file, DEVICE)
        tokenizer = CLIPTokenizer(tokenizer_name, merges_file=merges_file)
        clip = models["clip"]
        clip.to(device)
        print(tokenizer)
        # Convert into a list of length Seq_Len=77
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        print(cond_tokens)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        cond_context = clip(cond_tokens)
        # Convert into a list of length Seq_Len=77
        uncond_prompt=""
        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
        # (Batch_Size, Seq_Len)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        uncond_context = clip(uncond_tokens)
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
        context = torch.cat([cond_context, uncond_context])
        to_idle(clip)
        sampler = DDPMSampler(generator)
        print(sampler)
        sampler.set_inference_timesteps(n_inference_steps)

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        latents = torch.randn(latents_shape, generator=generator, device=device)
        diffusion = models["diffusion"]
        diffusion.to(device)
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            # batch size 4, lat_h, lat_w -> 2 * batch  size (1 with prompt &  without
            model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            # removed noise from the image
            latents = sampler.step(timestep, latents, model_output)
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
