# stable_diffusion_from_scratch
Implementing stable diffusiuon from scratch based on https://www.youtube.com/watch?v=ZBKpAp_6TGI&amp;t=8778s

To run this you need to download 3 files and insert it into your working directory:

vocab.json and merges.txt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer

v1-5-pruned-emaonly.ckpt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

From this project I gain a lot of knowledge: What is DDPM, what are the parts of Stable diffusion (CLIP, U-Net, autoencoders witl latent space, math under the DDPM, training and inference process).

My future work related to this work is implement telegram bor with API for that code.

Status: FastAPI endpoint is implemented, bot and docker not yet
