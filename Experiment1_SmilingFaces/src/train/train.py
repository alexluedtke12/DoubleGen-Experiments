#!/usr/bin/env python3

#######################################
# training config

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    which_wgt = None  # will be set by command line argument
    num_epochs = None  # will be set by command line argument
    lr_decay_every = None # will be set by command line argument
    save_individual_images_at = None  # will be set by command line argument
    image_size = 64  # the generated image resolution
    train_batch_size = None  # will be set by command line argument
    # train_batch_size = 64
    eval_batch_size = 42  # how many images to sample during evaluation
    learning_rate = 1e-4
    lr_warmup_epochs = 2
    gradient_accumulation_steps = 1
    # lr_warmup_steps = 500
    save_epochs = 25
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'CelebA-attrs-model'  # the model name locally and on the HF Hub
    subdir = None # will be set after command line arguments read in
    num_train_timesteps = 1000 # number of time steps to use during noising
    beta_schedule = 'squaredcos_cap_v2'
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

#######################################
# parse command line arguments and add them to config
import argparse

parser = argparse.ArgumentParser(description='Training script with configurable weight type')
parser.add_argument('--which_wgt', type=str, required=True,
                    help='Weight type to use for training')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='Number of epochs to train')
parser.add_argument('--lr_decay_every', type=int, required=True,
                    help='Cut lr in half every lr_decay_every epochs')
parser.add_argument('--save_individual_images_at', type=int, required=False, default=-1,
                    help='Save individual images at this epoch, in addition to at the final epoch. If not supplied then will only save individual images at final epoch')
parser.add_argument('--large_net', type=int, required=False, default=False,
                    help='Should the UNet be made large (for final, clean run) or smaller (for nuisance sensitivity studies)?')
args = parser.parse_args()

config.which_wgt = args.which_wgt
config.num_epochs = args.num_epochs
config.lr_decay_every = args.lr_decay_every
config.save_individual_images_at = args.save_individual_images_at
config.large_net = True if args.large_net==1 else False

config.train_batch_size = 32 if config.large_net else 64

config.subdir = f'{config.which_wgt}_large_net' if config.large_net else config.which_wgt

#######################################
# imports

from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

#######################################
# dataloader

def get_dataloader(batch_size:int=64,num_cpu=4):
    "Builds a set of dataloaders with a batch_size"

    config.dataset_name = "./CelebA-attrs"
    dataset = load_from_disk(config.dataset_name)
    dataset = dataset.remove_columns(['image','mask'])
    # dataset = dataset.select(range(1000))

    ####################
    # only keep rows with nonzero weight
    dataset = dataset.select(np.where(np.array(dataset[config.which_wgt]) != 0)[0])
    ####################

    class SquareCenterCrop:
        def __call__(self, img):
            w, h = img.size
            size = min(w, h)
            return transforms.CenterCrop(size)(img)

    normalize_transform = transforms.Normalize([0.5], [0.5])
    to_tensor = transforms.ToTensor()
    
    # Transforms that should affect both image and mask
    shared_preprocess = transforms.Compose([
        # transforms.RandomAffine(0,translate=(0,.03)),
        SquareCenterCrop(),
        transforms.RandomHorizontalFlip(),
    ])
    
    # RGB-specific transforms with BICUBIC interpolation
    rgb_preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Mask-specific transforms with NEAREST interpolation
    mask_preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Datasets offer a handy set_transform() method to apply the image transformations on the fly during training:
    def transform(examples):
        images = []
        masks = []
        for rgba_image in examples["image_and_mask"]:
            rgba_image = shared_preprocess(rgba_image)
            # rgb_image = to_tensor(rgba_image.convert("RGB"))
            rgb_image = rgb_preprocess(rgba_image.convert("RGB"))
            mask_image = mask_preprocess(rgba_image.getchannel("A"))

            images.append(rgb_image)
            masks.append(mask_image)
            
        return {"image": images, "mask": masks, "wgt": examples[config.which_wgt]}
    dataset.set_transform(transform)

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_cpu,
        persistent_workers=True,  # Keep worker processes alive between iterations
        pin_memory=True,         # Pin memory for faster CPU->GPU transfer
        )
    return train_dataloader

#######################################
# more imports

import matplotlib.pyplot as plt
import torch.nn.functional as F

#######################################
# for plotting
from PIL import Image
import math

def make_grid(images, rows, cols):
    images = images[:(rows*cols)]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

#######################################
# exponential moving average

from copy import deepcopy

def create_model_ema(model, decay=0.999):
    model_ema = deepcopy(model)
    for param in model_ema.parameters():
        param.requires_grad = False
    return model_ema

def update_ema(model, model_ema, decay=0.999):
    with torch.no_grad():
        for model_param, ema_param in zip(model.parameters(), model_ema.parameters()):
            ema_param.data = ema_param.data * decay + model_param.data * (1.0 - decay)

#######################################
# training function

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from pathlib import Path
import os

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, config.subdir, "logs") # updated 'logging_dir' to 'project_dir' as per https://github.com/huggingface/accelerate/issues/1559
    )
    
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(os.path.join(config.output_dir, config.subdir), exist_ok=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    train_dataloader = get_dataloader(config.train_batch_size,num_cpu=3)#num_cpu=min(cpu_count()-2,2*accelerator.num_processes))
    # train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    if config.large_net:
        boc = (256, 512, 1024)
    else:
        boc = (128, 384, 768)

    from diffusers import UNet2DModel
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=4,  
        block_out_channels=boc,  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim = 16,
    )
    model_ema = create_model_ema(model, decay=0.999)  # EMA model initialization
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # from diffusers.optimization import get_cosine_schedule_with_warmup
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #    optimizer=optimizer,
    #    num_warmup_steps=config.lr_warmup_steps,
    #    num_training_steps=(len(train_dataloader) * config.num_epochs),
    # )
    from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
    if config.lr_warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=int(config.lr_warmup_epochs*len(train_dataloader)))
        decay = ExponentialLR(optimizer, gamma=0.5 ** (1/(len(train_dataloader) * config.lr_decay_every)))
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[int(config.lr_warmup_epochs*len(train_dataloader))])
    else:
        lr_scheduler = ExponentialLR(optimizer, gamma=0.5 ** (1/(len(train_dataloader) * config.lr_decay_every)))
    
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule=config.beta_schedule)

    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    accelerator.wait_for_everyone()
    model, model_ema, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, model_ema, optimizer, train_dataloader, lr_scheduler
    )

    ############################
    # evaluation fun
    from diffusers import DDPMPipeline
    def evaluate(config, epoch, pipeline, out_dir):
        # Move pipeline to accelerator device
        pipeline = accelerator.prepare(pipeline)
        
        # Create images on each GPU
        if out_dir=='model_ema':
            run_extra = ((epoch == config.num_epochs - 1) or (epoch == config.save_individual_images_at - 1))
            batch_size = (3000 if run_extra else config.eval_batch_size) // accelerator.num_processes
        else:
            run_extra = False
            batch_size = config.eval_batch_size // accelerator.num_processes
        
        jj_ub = 4 # when run_extra is True, generate jj_ub*3000 images
        for jj in range(jj_ub if run_extra else 1):
            local_images = pipeline(
                batch_size=batch_size,
                generator=torch.manual_seed(jj_ub*(config.seed + accelerator.process_index) + jj),
            ).images
            
            # Gather images from all processes
            gathered_images = accelerator.gather_for_metrics(local_images, use_gather_object=True)
            
            if accelerator.is_main_process:
                # Flatten if needed (if gathered_images is a list of lists)
                if isinstance(gathered_images[0], list):
                    gathered_images = [img for sublist in gathered_images for img in sublist]
                test_dir = os.path.join(config.output_dir, config.subdir, out_dir, "samples")
                os.makedirs(test_dir, exist_ok=True)
                individual_dir = os.path.join(config.output_dir, config.subdir, out_dir, "individual_samples", f"epoch_{epoch:04d}")
                os.makedirs(individual_dir, exist_ok=True)

                # Save individual images
                for idx, image in enumerate(gathered_images):
                    image.save(os.path.join(individual_dir, f"sample_{(3000*jj+idx):04d}.png"))

                if jj==0: # only want to save image grid and log on the first loop. Otherwise, just want to save individual images (as above)
                    image_grid = make_grid(gathered_images, rows=6, cols=config.eval_batch_size//6)
                    image_grid.save(f"{test_dir}/{epoch:04d}.png")
            
                    # Log to tensorboard
                    writer = SummaryWriter(os.path.join(config.output_dir, config.subdir, "logs", "train_example", out_dir))
                    writer.add_image('', transforms.ToTensor()(image_grid), global_step=epoch)
                    writer.close()

    ############################
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        # accelerator.print('inside loop over epochs')
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        # without the line below the head GPU was sometimes failing to start training
        accelerator.wait_for_everyone()
        
        for step, batch in enumerate(train_dataloader):
            accelerator.wait_for_everyone()

            clean_images = batch['image']
            masks = batch['mask']

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                mask_extraweight = 8
                loss = (batch['wgt']*((1+mask_extraweight*masks)*F.mse_loss(noise_pred, noise, reduction='none')).mean(dim=(1,2,3))/(1+mask_extraweight*masks).mean(dim=(1,2,3))).mean()
                accelerator.backward(loss)

                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 4.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update EMA after each step
            update_ema(model, model_ema, decay=0.999)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "grad_norm": grad_norm.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if (epoch + 1) % config.save_epochs == 0 or epoch == config.num_epochs - 1 or epoch == config.save_individual_images_at - 1:
            # Clear cache before evaluation
            torch.cuda.empty_cache() 
            with torch.no_grad():
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                evaluate(config, epoch, pipeline, out_dir='model')
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                
                pipeline_ema = DDPMPipeline(unet=accelerator.unwrap_model(model_ema), scheduler=noise_scheduler)
                evaluate(config, epoch, pipeline_ema, out_dir='model_ema')
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                
            if accelerator.is_main_process:
                model_dir = os.path.join(config.output_dir, config.subdir, "model")
                os.makedirs(model_dir, exist_ok=True)
                pipeline.save_pretrained(model_dir)
                
                model_ema_dir = os.path.join(config.output_dir, config.subdir, "model_ema")
                os.makedirs(model_ema_dir, exist_ok=True)
                pipeline_ema.save_pretrained(model_ema_dir)
                
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)

############################
# Launch the training loop
if __name__ == "__main__":
    train_loop(config)