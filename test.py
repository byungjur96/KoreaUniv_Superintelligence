import os
import argparse
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel

DATASET_PATH = "/workspace/AAA740/Dataset"
CHECKPOINT_DIR = "/workspace/AAA740/checkpoints/{}"

EXPERIMENTS = ['Batch1', 'Batch4', 'RadBERT', 'SapBERT', 'zero_init', 'zero_init2', 'Vanila']

def run_with_bert(prompts,
                  CHECKPOINT_PATH,
                  SAVE_PATH,
                  prompt_type,
                  TEXT_ENCODER,
                  height = 512,  # default height of Stable Diffusion
                  width = 512,  # default width of Stable Diffusion
                  num_inference_steps = 50,  # Number of denoising steps
                  guidance_scale = 7.5,  # Scale for classifier-free guidance
                  seed=0,
                  tokenizer_max_length = 77
                  ):
    # Load Pretrained Model
    vae = AutoencoderKL.from_pretrained(CHECKPOINT_PATH, subfolder="vae", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    tokenizer.model_max_length = tokenizer_max_length
    text_encoder = AutoModel.from_pretrained(TEXT_ENCODER)
    unet = UNet2DConditionModel.from_pretrained(CHECKPOINT_PATH, subfolder="unet", use_safetensors=True)
    scheduler = PNDMScheduler.from_pretrained(CHECKPOINT_PATH, subfolder="scheduler")
    
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)
    
    if torch.cuda.is_available():
        generator = torch.cuda.manual_seed(seed)  # Seed generator to create the initial latent noise
    else:
        generator = torch.manual_seed(seed)  # Seed generator to create the initial latent noise

    for p in prompts:
        if prompt_type == "custom":
            prompt = p
        else:
            prompt = test_csv[p][1]
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = text_encoder(**text_input.to(torch_device))[0]
            
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer("", padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=torch_device,
        )

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        images = (image * 255).round().astype("uint8")
        image = Image.fromarray(image)
        file_name = test_csv[p][2] if not prompt_type == "custom" else f"sample_{prompt}.png"
        image.save(f"{SAVE_PATH}/{file_name}")

def run_with_pipeline(prompts, CHECKPOINT_PATH, SAVE_PATH, prompt_type, seed):
    if torch.cuda.is_available():
        generator = torch.cuda.manual_seed(seed)  # Seed generator to create the initial latent noise
    else:
        generator = torch.manual_seed(seed)  # Seed generator to create the initial latent noise
    pipe = StableDiffusionPipeline.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.float16,
        generator = generator, 
        safety_checker = None,  # To Prevent 'Potential NSFW content' Error
        use_safetensors=True
        )
    pipe = pipe.to("cuda")
    for p in prompts:
        if prompt_type == "custom":
            prompt = p
        else:
            prompt = test_csv[p][1]
        image = pipe(prompt).images[0]
        file_name = test_csv[p][2] if not prompt_type == "custom" else f"sample_{prompt}.png"
        image.save(f"{SAVE_PATH}/{file_name}")

def run_with_vanila(prompts, SAVE_PATH, prompt_type, seed):
    if torch.cuda.is_available():
        generator = torch.cuda.manual_seed(seed)  # Seed generator to create the initial latent noise
    else:
        generator = torch.manual_seed(seed)  # Seed generator to create the initial latent noise
    pipe = StableDiffusionPipeline.from_pretrained(
        'compvis/stable-diffusion-v1-4',
        torch_dtype=torch.float16,
        generator = generator, 
        safety_checker = None,  # To Prevent 'Potential NSFW content' Error
        use_safetensors=True
        )
    pipe = pipe.to("cuda")
    for p in prompts:
        if prompt_type == "custom":
            prompt = p
        else:
            prompt = test_csv[p][1]
        image = pipe(prompt).images[0]
        file_name = test_csv[p][2] if not prompt_type == "custom" else f"sample_{prompt}.png"
        image.save(f"{SAVE_PATH}/{file_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=int, default=-1)
    parser.add_argument('--save_dir', type=str, default="./result")
    parser.add_argument('--prompt', type=str, choices=["random", "all", "select", "custom"], default="random")
    parser.add_argument('--select', type=int, default=1, help="Select prompt id")
    parser.add_argument('--custom', type=str, default="Description of layers of the gastrointestinal tract including epithelium, lamina propria, muscularis mucosa, and smooth muscle.", help="Own Prompt as an Input")
    args = parser.parse_args()
    
    # Make Folder to Save Results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"The folder '{args.save_dir}' has been created.")
    else:
        print(f"The folder '{args.save_dir}' already exists.")
        
    # Load CSV file of test dataset
    with open(f'{DATASET_PATH}/test_dataset.csv', 'r') as file:
        csv_reader = csv.reader(file)
        global test_csv
        test_csv = []
        for row in csv_reader:
            test_csv.append(row)
    
    if 0 <= args.experiments < len(EXPERIMENTS):
        experiment_list = [EXPERIMENTS[args.experiments]]
    else:
        experiment_list = [EXPERIMENT for EXPERIMENT in EXPERIMENTS]
            
    if args.prompt == "random":
        prompt_num = [random.randint(1, len(test_csv) - 1)]
    elif args.prompt == "all":
        prompt_num = [i for i in range(1, len(test_csv))]
    elif args.prompt == "select":
        prompt_num = [args.select]
    else:
        prompt_num = [args.custom]
        
    for experiment in experiment_list:
        CHECKPOINT_PATH = CHECKPOINT_DIR.format(experiment)
        SAVE_PATH = f"{args.save_dir}/{experiment}"
        
        if not os.path.exists(f"{SAVE_PATH}"):
            os.makedirs(f"{SAVE_PATH}")
            print(f"The folder '{SAVE_PATH}' has been created.")
        else:
            print(f"The folder '{SAVE_PATH}' already exists.")
            
        if experiment == "RadBERT":
            run_with_bert(prompts=prompt_num, CHECKPOINT_PATH=CHECKPOINT_PATH, SAVE_PATH=SAVE_PATH,
                          prompt_type=args.prompt, TEXT_ENCODER="StanfordAIMI/RadBERT",seed=0)
        elif experiment == "SapBERT":
            run_with_bert(prompts=prompt_num, CHECKPOINT_PATH=CHECKPOINT_PATH, SAVE_PATH=SAVE_PATH,
                          prompt_type=args.prompt, TEXT_ENCODER="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", seed=0)
        elif experiment == "Vanila":
            run_with_vanila(prompts=prompt_num, SAVE_PATH=SAVE_PATH, prompt_type=args.prompt, seed=0)
        else:
            run_with_pipeline(prompts=prompt_num, CHECKPOINT_PATH=CHECKPOINT_PATH, SAVE_PATH=SAVE_PATH, prompt_type=args.prompt, seed=0)
        

if __name__ == "__main__":
    main()