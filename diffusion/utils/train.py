"""
common functions for train scripts
"""
import torch
from diffusers import ConsistencyDecoderVAE, DPMSolverMultistepScheduler, Transformer2DModel, AutoencoderKL
import wandb
import os
import json
from diffusers import PixArtSigmaPipeline
from diffusion.utils.logger_ac import get_logger
from diffusion.utils.text_embeddings import get_path_for_encoded_prompt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from diffusion.data.datasets.utils_ac import get_t5_feature_path
from .accelerate import wait_for_everyone
from diffusion.utils.dist_utils import flush
from diffusion.utils.image_evaluation import generate_images
from diffusion.utils.cmmd import get_cmmd_for_images

logger = get_logger(__name__)


def get_cmmd_samples(
        config,
):
    """
    returns list of cmmd sample groups,
    each group is a dict with 'items' and 'name'
    and each of those items is a pixart type item with 'path' and 'prompt'

    expects cmmd config to have 'sample_jsons' list of dicts, each with 'path' and 'name'
    """
    if not config.get('cmmd') or not config.cmmd.get('sample_jsons'):
        logger.info("No CMMD config provided. Skipping get_cmmd_samples")
        return []

    data_root = config.data_root
    sample_jsons = config.cmmd.get('sample_jsons')

    # deterministically sample image-text pairs from train and val sets
    def load_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def build_group(dict):
        items = load_json(os.path.join(data_root, dict['path']))
        return {
            'items': items,
            'name': dict['name'],
        }

    groups = [build_group(dict) for dict in sample_jsons]
    
    return groups

def log_cmmd(
        pipeline,
        accelerator,
        config,
        global_step,
        ):
    wait_for_everyone()
    if not accelerator.is_main_process:
        return
    
    samples = get_cmmd_samples(config=config)
    if not samples:
        logger.warning("No CMMD data provided. Skipping CMMD calculation.")
        return
    flush()
    
    data_root = config.data_root
    t5_save_dir = config.t5_save_dir
    max_token_length = config.max_token_length
    num_workers = config.num_workers
    
    # generate images using the text captions
    logger.info("Generating CMMD images using the text captions...")
   
    # extract saved t5 features and return 2 item tuple
    # of batch tensors for prompt embeds and attention masks
    def build_t5_batch_tensors_from_item_paths(paths):
        caption_feature_list = []
        attention_mask_list = []
        for item_path in paths:
            npz_path = get_t5_feature_path(
                t5_save_dir=t5_save_dir, 
                image_path=item_path,
                relative_root_dir=data_root,
                max_token_length=max_token_length,
                )
            # should this reuse or share logic with the dataset get item?
            embed_dict = np.load(npz_path)
            caption_feature = torch.from_numpy(embed_dict['caption_feature'])
            attention_mask = torch.from_numpy(embed_dict['attention_mask'])
            caption_feature_list.append(caption_feature)
            attention_mask_list.append(attention_mask)
        return torch.stack(caption_feature_list), torch.stack(attention_mask_list)

    # generate images and compute CMMD for either train or val items
    # note: generated images are squares of image_size
    def generate_images_and_cmmd(items):
        negative_prompt = config.cmmd.get('negative_prompt')
        should_encode_prompts = config.cmmd.get('should_encode_prompts', False)
        if not should_encode_prompts:
            caption_features, attention_masks = build_t5_batch_tensors_from_item_paths([item['path'] for item in items])
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
            prompts = None
            if negative_prompt:
                negative_prompt_embed_dict = torch.load(
                    get_path_for_encoded_prompt(prompt=negative_prompt, max_token_length=max_token_length),
                    map_location='cpu'
                    )
                repeats = caption_features.size(0)
                negative_prompt_embeds = negative_prompt_embed_dict['prompt_embeds'].unsqueeze(0).repeat(repeats, 1, 1)
                negative_prompt_attention_mask = negative_prompt_embed_dict['prompt_attention_mask'].unsqueeze(0).repeat(repeats, 1)
                negative_prompt = None
        else:
            prompts = [item['prompt'] for item in items]
            if negative_prompt is None:
                negative_prompt = ''
            caption_features = None
            attention_masks = None
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        generated_images = generate_images(
            pipeline=pipeline,
            accelerator=accelerator,
            prompt_embeds=caption_features,
            prompt_attention_mask=attention_masks,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            batch_size=config.cmmd.image_gen_batch_size,
            num_inference_steps=config.cmmd.num_inference_steps,
            width=config.image_size,
            height=config.image_size,
            guidance_scale=config.cmmd.guidance_scale,
            device=accelerator.device,
            max_token_length=config.max_token_length,
            prompts=prompts,
            negative_prompt=negative_prompt,
            num_workers=num_workers,
        )
        orig_image_paths = [os.path.join(data_root, item['path']) for item in items]
        # resize original images
        orig_images = [Image.open(image_path) for image_path in orig_image_paths]
        cmmd_score = get_cmmd_for_images(
            ref_images=orig_images,
            eval_images=generated_images,
            batch_size=config.cmmd.clip_batch_size,
            device=accelerator.device,
        )
        return generated_images, cmmd_score
    logger.info('generating images and cmmd scores...')

    for group in samples:
        name = group['name']
        items = group['items']
        logger.info(f'generating images and cmmd scores for group {name}...')
        generated_images, cmmd_score = generate_images_and_cmmd(items)

        for tracker in accelerator.trackers:
            max_images_logged = config.cmmd.get('max_images_logged', 10)
            if tracker.name == 'wandb':
                prompts = [item['prompt'] for item in items]
                wandb_images = [wandb.Image(
                    image,
                    caption=prompt,
                    ) for image, prompt in list(zip(generated_images, prompts))[:max_images_logged]]
                logger.info(f'logging {name} cmmd images to wandb...')
                tracker.log({
                    f"cmmd_gen_images_{name}": wandb_images, 
                    }, 
                    step=global_step,
                    )
            else:
                logger.warn(f"CMMD logging not implemented for {tracker.name}")

        logger.info(f'logging cmmd scores for group {name} to wandb...')
        accelerator.log({
                f"cmmd_score_{name}": cmmd_score,
            }, 
            step=global_step
        )

@torch.inference_mode()
def log_eval_images(
    accelerator,
    config,
    pipeline, 
    global_step,
    ):
    wait_for_everyone()
    if not (config.eval and 
            config.eval.batch_size and 
            config.eval.guidance_scale and
            config.eval.prompts
            ):
        logger.warning('No eval config provided. Skipping log evaluation images')
        return
    flush()
    logger.info(f"Generating {len(config.eval.prompts)} eval images... ")

    batch_size = config.eval.batch_size
    seed = config.eval.get('seed', 0)
    guidance_scale = config.eval.get('guidance_scale', 4.5)
    eval_sample_prompts = config.eval.prompts
    eval_negative_prompt = config.eval.get('negative_prompt')
    max_token_length = config.max_token_length
    num_workers = config.num_workers

    prompt_embeds_list = []
    prompt_attention_mask_list = []
    negative_prompt_embeds_list = []
    negative_prompt_attention_mask_list = []

    # kwarg placeholders:
    # assign embeds kwargs if not should_encode_prompts
    prompt_embeds = None
    prompt_attention_mask = None
    negative_prompt_embeds = None
    negative_prompt_attention_mask = None
    # assign prompts kwargs if should_encode_prompts
    prompts = None
    negative_prompt = None

    image_logs = []
    images = []

    should_encode_prompts = config.eval.get('should_encode_prompts', False)

    if not should_encode_prompts:
        for prompt in eval_sample_prompts:
            prompt_embed_dict = torch.load(
                    get_path_for_encoded_prompt(prompt, max_token_length),
                    map_location='cpu'
                    )
            prompt_embeds_list.append(prompt_embed_dict['prompt_embeds'])
            prompt_attention_mask_list.append(prompt_embed_dict['prompt_attention_mask'])

        if eval_negative_prompt:
            negative_prompt_embed_dict = torch.load(
                    get_path_for_encoded_prompt(eval_negative_prompt, max_token_length),
                    map_location='cpu'
                    )
            negative_prompt_embeds_list = [negative_prompt_embed_dict['prompt_embeds']] * len(eval_sample_prompts)
            negative_prompt_attention_mask_list = [negative_prompt_embed_dict['prompt_attention_mask']] * len(eval_sample_prompts)
    
        prompt_embeds = torch.stack(prompt_embeds_list)
        prompt_attention_mask = torch.stack(prompt_attention_mask_list)

        negative_prompt_embeds = torch.stack(negative_prompt_embeds_list) if negative_prompt_embeds_list else None
        negative_prompt_attention_mask = torch.stack(negative_prompt_attention_mask_list) if negative_prompt_attention_mask_list else None
    else:
        prompts = eval_sample_prompts
        negative_prompt = eval_negative_prompt

    images = generate_images(
        pipeline=pipeline,
        accelerator=accelerator,
        prompts=prompts,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt=negative_prompt,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        batch_size=batch_size,
        num_inference_steps=config.eval.num_inference_steps,
        width=config.image_size,
        height=config.image_size,
        seed=seed,
        guidance_scale=guidance_scale,
        device=accelerator.device,
        max_token_length=max_token_length,
        num_workers=num_workers,
    )

    flush()

    logger.info('finished generating eval images. logging...')
    for prompt, image in zip(eval_sample_prompts, images):
        image_logs.append({"prompt": prompt, "image": image})

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":  
            formatted_images = []
            for image_log in image_logs:
                image = image_log['image']
                prompt = image_log['prompt']
                image = wandb.Image(image, caption=prompt)
                formatted_images.append(image)
                    
            tracker.log(
                {"eval_images": formatted_images},
                step=global_step,
                )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    return image_logs


def prepare_for_inference(accelerator, model):
    # model = accelerator.unwrap_model(model)
    model.eval()
    return model

def prepare_for_training(accelerator, model):
    model = accelerator.prepare(model)
    model.train()
    return model

def get_step_bucket_loss(
        config,
        gathered_timesteps,
        gathered_losses,
        ):
    """
    Bucket loss by timestep, and log.
    5 buckets
    """
    num_buckets = 5
    bucket_ranges = np.linspace(0, config.train_sampling_steps, num=num_buckets + 1, endpoint=True).astype(int)
    bucket_losses = {i: [] for i in range(num_buckets)}
    mean_bucket_losses = {}
    # Accumulate losses in the respective buckets
    for i in range(num_buckets):
        mask = (gathered_timesteps >= bucket_ranges[i]) & (gathered_timesteps < bucket_ranges[i + 1])
        if np.any(mask):
            bucket_losses[i].extend(gathered_losses[mask])

    # Compute mean loss for each bucket and log
    for i in range(num_buckets):
        if bucket_losses[i]:
            bucket_mean_loss = np.mean(bucket_losses[i])
            # get boundaries of bucket for log
            lower_bound = bucket_ranges[i]
            upper_bound = bucket_ranges[i + 1] - 1  # Subtract 1 to make the range inclusive
            bucket_name = f"step_bucket_{lower_bound}-{upper_bound}"
            mean_bucket_losses[bucket_name] = bucket_mean_loss
            # logger.info(f"Global Step {global_step}: Bucket {i} Validation Loss: {bucket_mean_loss:.4f}")
            # accelerator.log({"validation_loss_bucket_{}".format(i): bucket_mean_loss}, step=global_step)
    return mean_bucket_losses
