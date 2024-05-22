import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image
from torch.utils.data import RandomSampler

# probably have to import my dataset so that build_dataset can find it?
from diffusion.data.datasets.InternalData_ms_ac import InternalDataMSSigmaAC
from diffusion.data.datasets.utils_ac import get_t5_feature_path

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.checkpoint_ac import load_checkpoint
from diffusion.utils.data_sampler_ac import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger_ac import get_logger
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

import wandb
import json
from diffusion.utils.text_embeddings import encode_prompts, get_path_for_encoded_prompt
from diffusion.utils.image_evaluation import generate_images, get_image_gen_pipeline
from diffusion.utils.cmmd import get_cmmd_for_images
from diffusion.model.nets.diffusers import convert_net_to_diffusers
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

warnings.filterwarnings("ignore")  # ignore warning

dtype_mapping = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
        'fp64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'no': torch.float32,
    }

def wait_for_everyone():
    # possible issue with accelerator.wait_for_everyone() breaking on linux kernel < 5.5
    # https://github.com/huggingface/accelerate/issues/1929
    synchronize()

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

@torch.inference_mode()
def log_eval_images(pipeline, global_step):
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

    prompt_embeds_list = []
    prompt_attention_mask_list = []
    negative_prompt_embeds_list = []
    negative_prompt_attention_mask_list = []

    image_logs = []
    images = []
    for prompt in eval_sample_prompts:
        prompt_embed_dict = torch.load(
                get_path_for_encoded_prompt(prompt, max_length),
                map_location='cpu'
                )
        prompt_embeds_list.append(prompt_embed_dict['prompt_embeds'])
        prompt_attention_mask_list.append(prompt_embed_dict['prompt_attention_mask'])

    if eval_negative_prompt:
        negative_prompt_embed_dict = torch.load(
                get_path_for_encoded_prompt(eval_negative_prompt, max_length),
                map_location='cpu'
                )
        negative_prompt_embeds_list = [negative_prompt_embed_dict['prompt_embeds']] * len(eval_sample_prompts)
        negative_prompt_attention_mask_list = [negative_prompt_embed_dict['prompt_attention_mask']] * len(eval_sample_prompts)
    
    prompt_embeds = torch.stack(prompt_embeds_list)
    prompt_attention_mask = torch.stack(prompt_attention_mask_list)

    negative_prompt_embeds = torch.stack(negative_prompt_embeds_list) if negative_prompt_embeds_list else None
    negative_prompt_attention_mask = torch.stack(negative_prompt_attention_mask_list) if negative_prompt_attention_mask_list else None

    images = generate_images(
        pipeline=pipeline,
        accelerator=accelerator,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
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

@torch.inference_mode()
def log_validation_loss(model, global_step):
    if not val_dataloaders or len(val_dataloaders) == 0:
        logger.warning("No validation data provided. Skipping validation.")
        return
    
    model.eval()
    logs = {}
    all_validation_losses = []
    step_bucket_losses = {} # key: bucket name, value: list of each batch's mean losses for that bucket. 
    logger.info(f"logging validation loss for {len(val_dataset)} images")
    for val_dataloader in val_dataloaders:
        dataset_validation_losses = []
        for batch in val_dataloader:
            # accelerate dataloader yields None when it is finished. need to break after that or get exception.
            # I think bc drop_last in batch sampler throws off the number of batches
            if batch is None or len(batch) == 0:
                logger.warning('log_validation_loss breaking after encountering empty batch.')
                break
            z = batch[0]
            latents = (z * config.scale_factor)

            y = batch[1]
            y_mask = batch[2]
            
            data_info = batch[3]

            # Sample multiple timesteps for each image
            bs = latents.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=latents.device).long()

            # Predict the noise residual and compute the validation loss
            with torch.no_grad():
                loss_term = train_diffusion.training_losses(
                    model, latents, timesteps, 
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                    )
                gathered_losses = accelerator.gather(loss_term['loss']).cpu().numpy()
                mean_loss = gathered_losses.mean()
                dataset_validation_losses.append(mean_loss)
                step_bucket_mean_loss = get_step_bucket_loss(
                    gathered_timesteps=accelerator.gather(timesteps).cpu().numpy(),
                    gathered_losses=gathered_losses,
                )
                # adds this batch's mean losses to each step bucket
                for bucket_name, bucket_loss in step_bucket_mean_loss.items():
                    if bucket_name not in step_bucket_losses:
                        step_bucket_losses[bucket_name] = []
                    step_bucket_losses[bucket_name].append(bucket_loss)

        # gather val losses across datasets
        all_validation_losses.extend(dataset_validation_losses)
        # mean val loss for dataset
        dataset_name = val_dataloader.dataset.name
        if dataset_name:
            dataset_validation_loss = np.mean(dataset_validation_losses)
            logs[f'val_loss_{bucket_name}'] = dataset_validation_loss
        
    # mean val loss across datasets
    mean_validation_loss = np.mean(all_validation_losses)
    logs['val_loss'] = mean_validation_loss

    # get mean of step bucket losses across batches and datasets
    for bucket_name, bucket_losses in step_bucket_losses.items():
        bucket_mean_loss = np.mean(bucket_losses)
        logs[f'val_loss_{bucket_name}'] = bucket_mean_loss
        
    info = f"Global Step {global_step}"
    info += ', '.join([f"{k}:{v:.4f}" for k, v in logs.items()])
    logger.info(info)
    accelerator.log(logs, step=global_step)

    model.train()

def get_cmmd_samples():
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
        global_step,
        ):
    wait_for_everyone()
    if not accelerator.is_main_process:
        return
    
    samples = get_cmmd_samples()
    if not samples:
        logger.warning("No CMMD data provided. Skipping CMMD calculation.")
        return
    flush()
    
    data_root = config.data_root
    t5_save_dir = config.t5_save_dir
    
    negative_prompt = config.cmmd.get('negative_prompt')

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
                max_token_length=max_length,
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
        caption_features, attention_masks = build_t5_batch_tensors_from_item_paths([item['path'] for item in items])
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if negative_prompt:
            negative_prompt_embed_dict = torch.load(
                get_path_for_encoded_prompt(negative_prompt, max_length),
                map_location='cpu'
                )
            repeats = caption_features.size(0)
            negative_prompt_embeds = negative_prompt_embed_dict['prompt_embeds'].unsqueeze(0).repeat(repeats, 1, 1)
            negative_prompt_attention_mask = negative_prompt_embed_dict['prompt_attention_mask'].unsqueeze(0).repeat(repeats, 1)

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

def prepare_for_inference(model):
    model = accelerator.unwrap_model(model)
    model.eval()
    return model

def prepare_for_training(model):
    model = accelerator.prepare(model)
    model.train()
    return model

def get_step_bucket_loss(
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

def train():
    global model
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1

    pipeline = None
    if config.eval.at_start or config.cmmd.at_start:
        model = prepare_for_inference(model)
        pipeline = _get_image_gen_pipeline(
            model=model,
            )

        if config.eval.at_start:
            log_eval_images(pipeline=pipeline, global_step=global_step)

        if config.cmmd.at_start:
            log_cmmd(pipeline=pipeline, global_step=global_step)
    
        model = prepare_for_training(model)
        del pipeline
        flush()
        logger.debug('finished w image gen pipeline')

    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        logger.debug('start epoch')
        data_time_start= time.time()
        data_time_all = 0
        for train_dataloader in train_dataloaders:
            for step, batch in enumerate(train_dataloader):
                if step < skip_step:
                    global_step += 1
                    continue    # skip data in the resumed ckpt
                # when using drop_last with batch sampler, we sometimes get empty batch
                if batch is None or len(batch) == 0:
                    # accelerate dataloader yields None when it is finished. need to break after that or get exception.
                    # I think bc drop_last in batch sampler throws off the number of batches
                    logger.warning('train break after encountering empty batch.')
                    break
                logger.debug('step: {}'.format(step))
                z = batch[0]
                clean_images = z * config.scale_factor

                y = batch[1]
                y_mask = batch[2]

                data_info = batch[3]

                # Sample a random timestep for each image
                bs = clean_images.shape[0]
                timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()

                grad_norm = None
                data_time_all += time.time() - data_time_start
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    logger.debug('accumulating')
                    optimizer.zero_grad()
                    loss_term = train_diffusion.training_losses(
                        model, clean_images, timesteps, 
                        model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                        )
                    loss = loss_term['loss'].mean()
                    logger.debug('accumulating backward')
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()

                gathered_losses = accelerator.gather(loss_term['loss']).detach().cpu().numpy()
                lr = lr_scheduler.get_last_lr()[0]
                
                mean_gathered_losses = gathered_losses.mean().item()
                logs = {'loss': mean_gathered_losses}
                dataset_name = train_dataloader.dataset.name
                if dataset_name:
                    logs.update({
                        f'{dataset_name}_loss': mean_gathered_losses,
                    })
                logs.update(lr=lr)
                # mean bucket losses
                step_bucket_mean_loss = get_step_bucket_loss(
                    gathered_timesteps=accelerator.gather(timesteps).cpu().numpy(),
                    gathered_losses=gathered_losses,
                )
                for bucket_name, bucket_loss in step_bucket_mean_loss.items():
                    logs[f'{bucket_name}_loss'] = bucket_loss
                if grad_norm is not None:
                    logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
                log_buffer.update(logs)

                if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                    t = (time.time() - last_tic) / config.log_interval
                    t_d = data_time_all / config.log_interval
                    avg_time = (time.time() - time_start) / (global_step - start_step + 1)
                    eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                    eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (steps_per_epoch - step - 1))))
                    log_buffer.average()
                    info = f"Global Step: {global_step}. Epochs/Total: {epoch}/{config.num_epochs}. Current Epoch Steps: {step + 1}/{steps_per_epoch}. total_eta: {eta}, " \
                        f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                    info += f's:({model.module.h}, {model.module.w}), ' if hasattr(model, 'module') else f's:({model.h}, {model.w}), '                
                    info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                    logger.info(info)
                    accelerator.log(log_buffer.output, step=global_step)
                    last_tic = time.time()
                    log_buffer.clear()
                    data_time_all = 0
                    
                global_step += 1
                data_time_start = time.time()

                # STEP END actions: save, log val loss, eval images, cmmd
                if config.save_model_steps and global_step % config.save_model_steps == 0:
                    save_state(
                        global_step=global_step,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                    )
                if config.log_val_loss_steps and global_step % config.log_val_loss_steps == 0:
                    log_validation_loss(model=model, global_step=global_step)
                
                should_log_eval = (config.eval.every_n_steps and global_step % config.eval.every_n_steps == 0)
                should_log_cmmd = (config.cmmd.every_n_steps and global_step % config.cmmd.every_n_steps == 0)

                if (should_log_eval or should_log_cmmd):
                    model = prepare_for_inference(model)
                    pipeline = _get_image_gen_pipeline(
                        model=model,
                        )
                    if should_log_eval:
                        log_eval_images(pipeline=pipeline, global_step=global_step)
                    
                    if should_log_cmmd:
                        log_cmmd(pipeline=pipeline, global_step=global_step)
                    
                    model = prepare_for_training(model)
                    del pipeline
                    flush()

        # EPOCH END actions: save, log val loss, eval images, cmmd
        if (config.save_model_epochs and epoch % config.save_model_epochs == 0) or epoch == config.num_epochs:
            save_state(
                global_step=global_step,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        if (config.log_val_loss_epochs and epoch % config.log_val_loss_epochs == 0) or epoch == config.num_epochs:
            log_validation_loss(model=model, global_step=global_step)
        
        should_log_eval = (config.eval.every_n_epochs and epoch % config.eval.every_n_epochs == 0) or \
            (epoch == config.num_epochs and config.eval.at_end)
        should_log_cmmd = (config.cmmd.every_n_epochs and epoch % config.cmmd.every_n_epochs == 0) or \
            (epoch == config.num_epochs and config.cmmd.at_end)

        if (should_log_eval or should_log_cmmd):
            model = prepare_for_inference(model)
            pipeline = _get_image_gen_pipeline(
                model=model,
                )
            if should_log_eval:
                log_eval_images(pipeline=pipeline, global_step=global_step)
            
            if should_log_cmmd:
                log_cmmd(pipeline=pipeline, global_step=global_step)
            
            model = prepare_for_training(model)
            del pipeline
            flush()

def _get_image_gen_pipeline(model):
    diffusers_transformer = convert_net_to_diffusers(
        state_dict=model.state_dict(),
        image_size=image_size,
    )
    diffusers_transformer = diffusers_transformer.to(torch_dtype)
    return get_image_gen_pipeline(
                pipeline_load_from=config.pipeline_load_from,
                torch_dtype=torch_dtype,
                device=accelerator.device,
                transformer=diffusers_transformer,
                )

def save_state(global_step, epoch, model, optimizer, lr_scheduler):
    wait_for_everyone()
    if accelerator.is_main_process:
        os.umask(0o000)
        save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler
                        )

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    torch_dtype = dtype_mapping[accelerator.mixed_precision]

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training. accelerator.num_processes: {accelerator.num_processes}")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None

    logger.info(f"vae scale factor: {config.scale_factor}")

    # null embed needed for: conditional dropout, eval, cmmd, and for the model class_dropout_prob (via load_checkpoint)
    if (config.eval or config.resume_from) and accelerator.is_main_process:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        eval_config = config.eval
        # load checkpoint, log eval images use null embed
        eval_prompts = eval_config.prompts or []
        eval_negative_prompt = eval_config.get('negative_prompt')
        cmmd_negative_prompt = config.cmmd.get('negative_prompt')
        prompts = [*eval_prompts, '']
        if eval_negative_prompt:
            prompts.append(eval_negative_prompt)
        if cmmd_negative_prompt:
            prompts.append(cmmd_negative_prompt)
        encode_prompts(
            prompts=prompts,
            pipeline_load_from=eval_config.pipeline_load_from,
            device=accelerator.device,
            batch_size=eval_config.batch_size,
            max_token_length=config.max_token_length,
        )
        
    model_kwargs = {"pe_interpolation": config.pe_interpolation, "config": config,
                    "model_max_length": max_length, "qk_norm": config.qk_norm,
                    "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.load_from is not None:
        config.load_from = args.load_from
    if config.load_from is not None:
        logger.info(f'loading checkpoint from {config.load_from}')
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    train_data = config.train_data
    val_data = config.val_data

    # build dataloader
    set_data_root(config.data_root)

    train_datasets = [
        build_dataset(
            data_dict, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
            null_embed_path=get_path_for_encoded_prompt('', max_length),
            max_length=max_length, config=config,
        ) for data_dict in train_data
    ]
    train_dataloaders = []
    # dataset = build_dataset(
    #     train_data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
    #     # real_prompt_ratio=config.real_prompt_ratio,
    #     null_embed_path=get_path_for_encoded_prompt('', max_length),
    #     max_length=max_length, config=config,
    # )
    
    val_datasets = None
    val_dataloaders = None

    if config.val_data:
        val_datasets = [
            build_dataset(
                data_dict, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
                max_length=max_length, config=config,
            ) for data_dict in val_data
        ]
        val_dataloaders = []
        # val_dataset = build_dataset(
        #     val_data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
        #     max_length=max_length, config=config,
        # )
    if config.multi_scale:
        
        for dataset in train_datasets:
            batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                    batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                    ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
            train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
            train_dataloaders.append(train_dataloader)

        if val_datasets:
            for val_dataset in val_datasets:
                val_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(val_dataset), dataset=val_dataset,
                                                    batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                    ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
                val_dataloader = build_dataloader(val_dataset, batch_sampler=val_batch_sampler, num_workers=config.num_workers)
                val_dataloaders.append(val_dataloader)

    else:
        for dataset in train_datasets:
            train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
            train_dataloaders.append(train_dataloader)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    
    # lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)
    # shouldnt need train_dataloader for build lr scheduler. if i use cosine instead of constant
    # may need to update with num steps since that is what it actually wants
    lr_scheduler = build_lr_scheduler(
        config=config, 
        optimizer=optimizer, 
        lr_scale_ratio=lr_scale_ratio,
        train_dataloader=None, # noop for constant lr
        )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(config.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step or 0
    # total_steps = len(train_dataloader) * config.num_epochs
    steps_per_epoch = sum(len(dataloader) for dataloader in train_dataloaders)
    total_steps =  steps_per_epoch * config.num_epochs

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        logger.info(f"resuming from checkpoint {config.resume_from['checkpoint']}")
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    train_dataloaders = [accelerator.prepare(train_dataloader) for train_dataloader in train_dataloaders]
    val_dataloaders = [accelerator.prepare(val_dataloader) for val_dataloader in val_dataloaders] if val_dataloaders else None
    train()
