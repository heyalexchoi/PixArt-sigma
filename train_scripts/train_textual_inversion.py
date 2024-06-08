"""
main differences

T5 text encoder will stay

everything else frozen

load data the same

evals the same

should load checkpoints and be able to convert to diffusers for image gen

or just do everything as diffusers
and convert my checkpoint at beginning


"""


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
from diffusion.utils.train import log_eval_images, log_cmmd, prepare_for_inference, \
    prepare_for_training, clamp_grad_norm_
        # , get_step_bucket_loss

from diffusion.model.nets.diffusers import convert_net_to_diffusers
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

warnings.filterwarnings("ignore")  # ignore warning

from diffusion.utils.accelerate import dtype_mapping, wait_for_everyone, set_fsdp_env

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from packaging import version
from torch.utils.data import Dataset
import safetensors

from torch.utils.data import DataLoader
import math
import torch.nn.functional as F

logger = get_logger(__name__)

class TextualInversionDataset(InternalDataMSSigmaAC):
    def __init__(
        self,
        tokenizer, 
        *args,
        **kwargs,
    ):
        super().__init__(
            load_t5_feat=False, 
            *args, **kwargs,
            )
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        img, text, _, data_info = super().__getitem__(index)

        # example = {
        #     'text': text,
        # }
        # quick and dirty here. if this works, should init w placeholder token
        # text = 'a photo of @reneeherbert_'
        # example["input_ids"] = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_token_length,
        #     return_tensors="pt",
        # ).input_ids[0]

        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt",
        )
        # squeeze to get rid of extra batch dimension
        input_ids = tokenized_text.input_ids.squeeze(0)
        attention_mask = tokenized_text.attention_mask.squeeze(0)
        return img, input_ids, attention_mask, data_info


@torch.inference_mode()
def log_validation_loss(
    text_encoder,
    diffuser,
    global_step,
    val_dataloaders,
    ):
    if not val_dataloaders or len(val_dataloaders) == 0:
        logger.warning("No validation data provided. Skipping validation.")
        return
    
    text_encoder.eval()
    logs = {}
    all_validation_losses = []
    # step_bucket_losses = {} # key: bucket name, value: list of each batch's mean losses for that bucket. 
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

            # get tokenized text / input_ids
            input_ids = batch[1].to(device=text_encoder.device)
            # encode input_ids w/ text_encoder
            y = text_encoder(input_ids)[0].to(dtype=torch_dtype)
            # very strange but seems like this shape is expected. assertion thrown otherwise
            y = y.unsqueeze(1)
            y_mask = batch[2]
            
            data_info = batch[3]

            # Sample multiple timesteps for each image
            bs = latents.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=latents.device).long()

            # Predict the noise residual and compute the validation loss
            with torch.no_grad():
                loss_term = train_diffusion.training_losses(
                    diffuser, latents, timesteps, 
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                    )
                gathered_losses = accelerator.gather(loss_term['loss']).cpu().numpy()
                mean_loss = gathered_losses.mean()
                dataset_validation_losses.append(mean_loss)
                # step_bucket_mean_loss = get_step_bucket_loss(
                #     config=config,
                #     gathered_timesteps=accelerator.gather(timesteps).cpu().numpy(),
                #     gathered_losses=gathered_losses,
                # )
                # adds this batch's mean losses to each step bucket
                # for bucket_name, bucket_loss in step_bucket_mean_loss.items():
                #     if bucket_name not in step_bucket_losses:
                #         step_bucket_losses[bucket_name] = []
                #     step_bucket_losses[bucket_name].append(bucket_loss)

        # gather val losses across datasets
        all_validation_losses.extend(dataset_validation_losses)
        # mean val loss for dataset
        dataset_name = val_dataloader.dataset.name
        if dataset_name:
            dataset_validation_loss = np.mean(dataset_validation_losses)
            logs[f'val_loss_{dataset_name}'] = dataset_validation_loss
        
    # mean val loss across datasets
    mean_validation_loss = np.mean(all_validation_losses)
    logs['val_loss'] = mean_validation_loss

    # get mean of step bucket losses across batches and datasets
    # for bucket_name, bucket_losses in step_bucket_losses.items():
    #     bucket_mean_loss = np.mean(bucket_losses)
    #     logs[f'val_loss_{bucket_name}'] = bucket_mean_loss
        
    info = f"Global Step {global_step}"
    info += ', '.join([f"{k}:{v:.4f}" for k, v in logs.items()])
    logger.info(info)
    accelerator.log(logs, step=global_step)

    text_encoder.train()


def train(
        accelerator,
        placeholder_token_ids,
        text_encoder,
        tokenizer,
        diffuser,
        train_dataloaders,
        val_dataloaders,
        optimizer,
        lr_scheduler,
        config,
        ):

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    if config.get('debug_nan', False):
        DebugUnderflowOverflow(text_encoder)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    pipeline = None
    if config.eval.at_start or config.cmmd.at_start:
        text_encoder = prepare_for_inference(
            accelerator=accelerator,
            model=text_encoder,
            )
        pipeline = _get_image_gen_pipeline(
            diffuser=diffuser,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            )
        logger.info(f'pipeline.text_encoder == text_encoder: {pipeline.text_encoder == text_encoder}')

        if config.eval.at_start:
            log_eval_images(
                accelerator=accelerator,
                config=config,
                pipeline=pipeline, 
                global_step=global_step
                )

        if config.cmmd.at_start:
            log_cmmd(
                pipeline=pipeline, 
                accelerator=accelerator,
                config=config,
                global_step=global_step,
                )
    
        text_encoder = prepare_for_training(
            accelerator=accelerator,
            model=text_encoder,
            )
        del pipeline
        flush()
        logger.debug('finished w image gen pipeline')

    # selector for non-target input embeddings that we will prevent from udpating
    input_embeddings = text_encoder.get_input_embeddings()
    index_no_updates = torch.ones((input_embeddings.weight.size(0),), dtype=torch.bool)
    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

    # train loop
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        logger.debug(f'start epoch {epoch}')
        data_time_start= time.time()
        data_time_all = 0
        for train_dataloader in train_dataloaders:
            logger.debug(f'train_dataloader.dataset.name: {train_dataloader.dataset.name} len: {len(train_dataloader)}')
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
                image_latents = z * config.scale_factor

                # get tokenized text / input_ids
                input_ids = batch[1].to(device=text_encoder.device)
                # encode input_ids w/ text_encoder
                y = text_encoder(input_ids)[0].to(dtype=torch_dtype)
                # very strange but seems like this shape is expected. assertion thrown otherwise
                y = y.unsqueeze(1)
                y_mask = batch[2]

                data_info = batch[3]

                logger.debug(f'batch len z: {len(z)}\n'
                            f'batch aspect_ratio: {data_info["aspect_ratio"]}\n'
                            )

                # Sample a random timestep for each image
                bsz = image_latents.shape[0]
                timesteps = torch.randint(0, config.train_sampling_steps, (bsz,), device=image_latents.device).long()

                grad_norm = None
                data_time_all += time.time() - data_time_start
                with accelerator.accumulate(text_encoder):
                    # Predict the noise residual
                    logger.debug('accumulating')
                    optimizer.zero_grad()
                    loss_term = train_diffusion.training_losses(
                        diffuser, image_latents, timesteps, 
                        model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                        )
                    loss = loss_term['loss'].mean()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        # grad_norm = accelerator.clip_grad_norm_(
                        #     text_encoder.get_input_embeddings().parameters(), 
                        #     config.gradient_clip,
                        #     )
                        # zero gradients for non-placeholder token embeddings
                        input_embeddings.weight.grad[index_no_updates] = 0
                        grad_norm = clamp_grad_norm_(
                            input_embeddings.parameters(),
                            min_norm=config.get('min_grad_norm', 0.0),
                            max_norm=config.gradient_clip,
                            )

                    optimizer.step()
                    lr_scheduler.step()


                logs = {}
                # zero the grads before optimizer step instead
                # Let's make sure we don't update any embedding weights besides the newly added token
                # selection of all input embeddings that defaults to True
                # index_no_updates = torch.ones((text_encoder.get_input_embeddings().weight.size(0),), dtype=torch.bool)
                # # set the placeholder token ids to False
                # index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                # with torch.no_grad():
                    # restore all non-placeholder token embeddings to original values
                    # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    #     index_no_updates
                    # ] = orig_embeds_params[index_no_updates]

                # log changes. MSE between current embeddings and original embeddings for both target and non-target
                def compare_embeddings(current_embeds, original_embeds, mask):
                    # Calculate the mean squared difference for the given mask
                    diff = current_embeds[mask] - original_embeds[mask]
                    mse = (diff ** 2).mean().item()
                    return mse

                mse_placeholder = compare_embeddings(input_embeddings.weight, orig_embeds_params, ~index_no_updates)
                mse_no_updates = compare_embeddings(input_embeddings.weight, orig_embeds_params, index_no_updates)
                logs.update({
                    'mse_placeholder': mse_placeholder,
                    'mse_no_updates': mse_no_updates,
                })
                # end log changes
                
                gathered_losses = accelerator.gather(loss).detach().cpu().numpy()
                lr = lr_scheduler.get_last_lr()[0]
                
                mean_gathered_losses = gathered_losses.mean().item()
                logs.update({'loss': mean_gathered_losses})
                dataset_name = train_dataloader.dataset.name
                if dataset_name:
                    logs.update({
                        f'{dataset_name}_loss': mean_gathered_losses,
                    })
                logs.update(lr=lr)
                # mean bucket losses
                # step_bucket_mean_loss = get_step_bucket_loss(
                #     config=config,
                #     gathered_timesteps=accelerator.gather(timesteps).cpu().numpy(),
                #     gathered_losses=gathered_losses,
                # )
                # for bucket_name, bucket_loss in step_bucket_mean_loss.items():
                #     logs[f'{bucket_name}_loss'] = bucket_loss
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
                    info += f's:({diffuser.module.h}, {diffuser.module.w}), ' if hasattr(diffuser, 'module') else f's:({diffuser.h}, {diffuser.w}), '                
                    info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                    logger.info(info)
                    accelerator.log(log_buffer.output, step=global_step)
                    last_tic = time.time()
                    log_buffer.clear()
                    data_time_all = 0
                    
                global_step += 1
                progress_bar.update(1)
                data_time_start = time.time()

                # STEP END actions: save, log val loss, eval images, cmmd
                if config.save_model_steps and global_step % config.save_model_steps == 0:
                    save_progress(
                        global_step=global_step,
                        text_encoder=text_encoder,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        placeholder_token_ids=placeholder_token_ids,
                        accelerator=accelerator,
                        args=args,
                    )
                if config.log_val_loss_steps and global_step % config.log_val_loss_steps == 0:
                    log_validation_loss(
                        text_encoder=text_encoder,
                        diffuser=diffuser,
                        global_step=global_step,
                        val_dataloaders=val_dataloaders,
                        )
                
                should_log_eval = (config.eval.every_n_steps and global_step % config.eval.every_n_steps == 0)
                should_log_cmmd = (config.cmmd.every_n_steps and global_step % config.cmmd.every_n_steps == 0)

                if (should_log_eval or should_log_cmmd):
                    text_encoder = prepare_for_inference(
                                    accelerator=accelerator,
                                    model=text_encoder,
                                    )
                    pipeline = _get_image_gen_pipeline(
                        diffuser=diffuser,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        )
                    if should_log_eval:
                        log_eval_images(
                            accelerator=accelerator,
                            config=config,
                            pipeline=pipeline, 
                            global_step=global_step
                            )
                    
                    if should_log_cmmd:
                        log_cmmd(
                            pipeline=pipeline, 
                            accelerator=accelerator,
                            config=config,
                            global_step=global_step,
                            )
                    
                    text_encoder = prepare_for_training(
                        accelerator=accelerator,
                        model=text_encoder,
                    )
                    del pipeline
                    flush()

        # EPOCH END actions: save, log val loss, eval images, cmmd
        if (config.save_model_epochs and epoch % config.save_model_epochs == 0) or epoch == config.num_epochs:
            save_progress(
                        global_step=global_step,
                        text_encoder=text_encoder,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        placeholder_token_ids=placeholder_token_ids,
                        accelerator=accelerator,
                        args=args,
                    )
        if (config.log_val_loss_epochs and epoch % config.log_val_loss_epochs == 0) or epoch == config.num_epochs:
            log_validation_loss(
                        text_encoder=text_encoder,
                        diffuser=diffuser,
                        global_step=global_step,
                        val_dataloaders=val_dataloaders,
                        )
        
        should_log_eval = (config.eval.every_n_epochs and epoch % config.eval.every_n_epochs == 0) or \
            (epoch == config.num_epochs and config.eval.at_end)
        should_log_cmmd = (config.cmmd.every_n_epochs and epoch % config.cmmd.every_n_epochs == 0) or \
            (epoch == config.num_epochs and config.cmmd.at_end)

        if (should_log_eval or should_log_cmmd):
            text_encoder = prepare_for_inference(
                                accelerator=accelerator,
                                model=text_encoder,
                                )

            pipeline = _get_image_gen_pipeline(
                diffuser=diffuser,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                )

            if should_log_eval:
                log_eval_images(
                    accelerator=accelerator,
                    config=config,
                    pipeline=pipeline, 
                    global_step=global_step
                    )
            
            if should_log_cmmd:
                log_cmmd(
                    pipeline=pipeline, 
                    accelerator=accelerator,
                    config=config,
                    global_step=global_step,
                    )
            
            text_encoder = prepare_for_training(
                            accelerator=accelerator,
                            model=text_encoder,
                            )
            del pipeline
            flush()

def load_ti_checkpoint(file_path):
    logger.info(f'loading ti checkpoint from {file_path}')
    with safetensors.safe_open(file_path, framework="pt") as f:
        if len(f.keys()) > 1:
            raise ValueError(f"Expected only one key in the checkpoint file, but found {len(f.keys())} keys.")
        data_dict = {}
        for key in f.keys():
            data_dict[key] = f.get_tensor(key)
            logger.info(f'Loaded {key} with shape {data_dict[key].shape}')
    # load into text encoder
    token = next(iter(data_dict.keys()))
    # token_id = tokenizer.convert_tokens_to_ids(token)
    embedding = data_dict[token]
    if embedding.shape[0] > 1:
        logger.warn(f'More than 1 embedding found for token {token}. Using the first one.')
        embedding = embedding[0]

    return token, embedding

def initialize_placeholder_token(args, tokenizer, text_encoder):
    # Add the placeholder token in tokenizer
    placeholder_token = args.placeholder_token
    initializer_token = args.initializer_token
    logger.info(f'initializing placeholder_token: {placeholder_token} using initializer_token: {initializer_token}')
    # get embedding for initializer token
    init_token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    
    # Check if initializer_token is a single token or a sequence of tokens
    if len(init_token_ids) != 1:
        raise ValueError(f"The initializer token must be a single token. Got {len(init_token_ids)} tokens.")
    
    if init_token_ids == tokenizer.unk_token_id:
        raise ValueError(f'The tokenizer does not know initializer token {initializer_token}. Please pass a different initializer_token.')
    
    initializer_token_id = init_token_ids[0]
    logger.info(f'initializer_token_id: {initializer_token_id}')
    token_embeds = text_encoder.get_input_embeddings().weight.data
    initial_embedding = token_embeds[initializer_token_id].clone()
    return placeholder_token, initial_embedding


def add_placeholder_token_and_embedding(token, embedding, tokenizer, text_encoder):
    # check if this token exists
    prospective_token_id = tokenizer.convert_tokens_to_ids(token)
    
    if prospective_token_id != tokenizer.unk_token_id:
        raise ValueError(f'The tokenizer already contains token {token}. Please pass a different placeholder_token.')
    
    # add the new token to tokenizer
    num_added_tokens = tokenizer.add_tokens([token])

    if num_added_tokens != 1:
        raise ValueError(f"The tokenizer already contained the token {token}. Please pass a different `placeholder_token` that is not already in the tokenizer.")
    
    logger.info(f"Token '{token}' passed checks and added to tokenizer.")
    
    placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

    # resize text encoder if needed
    if placeholder_token_id >= text_encoder.shared.num_embeddings:
        # Resize the token embeddings in the text_encoder to accommodate the new token
        logger.info('Resizing token embeddings in text_encoder to accommodate new token...')
        text_encoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    else:
        logger.info('text_encoder already has enough token embeddings to accommodate new token.')
    
    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        token_embeds[placeholder_token_id] = torch.tensor(embedding.clone(), dtype=text_encoder.shared.weight.dtype)
        logger.info(f"Embedding at {placeholder_token_id} assigned to token '{token}'.")

    return placeholder_token_id


def _get_image_gen_pipeline(
        diffuser, 
        text_encoder, 
        tokenizer,
        ):
    diffusers_transformer = convert_net_to_diffusers(
        state_dict=diffuser.state_dict(),
        image_size=image_size,
    )
    diffusers_transformer = diffusers_transformer.to(torch_dtype)
    return get_image_gen_pipeline(
                pipeline_load_from=config.pipeline_load_from,
                torch_dtype=torch_dtype,
                device=accelerator.device,
                transformer=diffusers_transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                )

def save_progress(
        text_encoder, 
        placeholder_token_ids, 
        accelerator, 
        args, 
        global_step,
        optimizer,
        lr_scheduler,
        safe_serialization=True,
        ):
    logger.info("Saving embeddings")
    wait_for_everyone()
    if accelerator.is_main_process:
        ti_dir = os.path.join(config.work_dir, 'textual_inversions')
        save_path = os.path.join(ti_dir,
                                f"embedding_{config.get('ti_subject', '')}_{global_step}.safetensors")
        save_state_path = os.path.join(ti_dir,
                                f"state_{config.get('ti_subject', '')}_{global_step}.pt")
        if not os.path.exists(ti_dir):
            os.makedirs(ti_dir, exist_ok=True)
        learned_embeds = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
        )
        learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, save_path)
        
        state_dict = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        torch.save(state_dict, save_state_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='diffuser checkpoint path to resume from')
    parser.add_argument('--ti_resume_from', help='ti checkpoint path to resume from')
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
    parser.add_argument("--placeholder_token", type=str, required=True)
    parser.add_argument("--initializer_token", type=str)
    parser.add_argument("--num_vectors", type=int, default=1, choices=[1], help="Only 1 vector seems to work in T5")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Debug log level",
    )
    args = parser.parse_args()

    if args.ti_resume_from is None and args.initializer_token is None:
        raise ValueError("initializer_token is required when ti_resume_from is not provided.")
    
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
    if args.ti_resume_from is not None and not os.path.exists(args.ti_resume_from):
        raise ValueError(f"ti_resume_from path {args.ti_resume_from} does not exist.")
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2

    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
        config.log_interval = 1
        # TODO: attach stream handler to root logger and set log level debug

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

    logger.info(f"placeholder_token: {args.placeholder_token} num_vectors: {args.num_vectors}")

    # null embed needed for: conditional dropout, eval, cmmd, and for the model class_dropout_prob (via load_checkpoint)
    if (config.eval or config.resume_from) and accelerator.is_main_process:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        eval_config = config.eval
        # load checkpoint, log eval images use null embed
        # eval_prompts = eval_config.prompts or []
        eval_prompts = [] # no point in pre-encoding prompts if we are not using them
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
    
    train_datasets = [
        TextualInversionDataset(
            # tokenizer=tokenizer,
            tokenizer=None, # assign this after tokenizer made
            resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
            null_embed_path=get_path_for_encoded_prompt('', max_length),
            max_length=max_length,
            **data_dict,
        ) for data_dict in train_data
    ]
    train_dataloaders = []    
    
    val_datasets = None
    val_dataloaders = None

    if config.val_data:
        val_datasets = [
            TextualInversionDataset(
                # tokenizer=tokenizer,
                tokenizer=None, # assign this after tokenizer made
                resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
                null_embed_path=get_path_for_encoded_prompt('', max_length),
                max_length=max_length,
                **data_dict,
            ) for data_dict in val_data
        ]
        val_dataloaders = []

    if config.multi_scale:
        
        for dataset in train_datasets:
            # not dropping last here
            batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                    batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                    ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
            train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
            train_dataloaders.append(train_dataloader)

        if val_datasets:
            for val_dataset in val_datasets:
                # not dropping last here
                val_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(val_dataset), dataset=val_dataset,
                                                    batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=False,
                                                    ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
                val_dataloader = build_dataloader(val_dataset, batch_sampler=val_batch_sampler, num_workers=config.num_workers)
                val_dataloaders.append(val_dataloader)

    else:
        for dataset in train_datasets:
            train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
            train_dataloaders.append(train_dataloader)

    # build optimizer and lr scheduler
    # lr_scale_ratio = 1
    # if config.get('auto_lr', None):
    #     lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
    #                                    config.optimizer, **config.auto_lr)
    # # optimizer = build_optimizer(model, config.optimizer)
    
    # lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)
    # shouldnt need train_dataloader for build lr scheduler. if i use cosine instead of constant
    # may need to update with num steps since that is what it actually wants
    # lr_scheduler = build_lr_scheduler(
    #     config=config, 
    #     optimizer=optimizer, 
    #     lr_scale_ratio=lr_scale_ratio,
    #     train_dataloader=None, # noop for constant lr
    #     )

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

    logger.info(f'steps_per_epoch: {steps_per_epoch}, total_steps: {total_steps}')

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        logger.info(f"resuming from checkpoint {config.resume_from['checkpoint']}")
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                #  optimizer=optimizer,
                                                #  lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    pipeline_name = 'PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers'
    
    tokenizer = T5Tokenizer.from_pretrained(pipeline_name, subfolder="tokenizer")

    for dataset in train_datasets + (val_datasets or []):
        dataset.tokenizer = tokenizer
    
    text_encoder = T5EncoderModel.from_pretrained(
        pipeline_name, subfolder="text_encoder", 
        torch_dtype=torch_dtype).to(accelerator.device)
    
    if args.ti_resume_from is not None:
        placeholder_token, init_embedding = load_ti_checkpoint(
            file_path=args.ti_resume_from,
        )
    else:
        placeholder_token, init_embedding = initialize_placeholder_token(
            args=args,
            tokenizer=tokenizer, 
            text_encoder=text_encoder,
            )

    placeholder_token_id = add_placeholder_token_and_embedding(
        token=placeholder_token,
        embedding=init_embedding,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        )

    diffuser = model
    # freeze diffuser
    diffuser.requires_grad_(False)
    
    # freeze text encoder
    for name, param in text_encoder.named_parameters():
        param.requires_grad = False
    # unfreeze the token embeddings parameters
    text_encoder.get_input_embeddings().weight.requires_grad = True

    if config.grad_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        diffuser.train()
        text_encoder.gradient_checkpointing_enable()
        # diffuser grad checkpointing is setup in build_model
        
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(steps_per_epoch / config.gradient_accumulation_steps)
    max_train_steps = config.num_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    diffuser = accelerator.prepare(diffuser)
    text_encoder = accelerator.prepare(text_encoder)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    train_dataloaders = [accelerator.prepare(train_dataloader) for train_dataloader in train_dataloaders]
    val_dataloaders = [accelerator.prepare(val_dataloader) for val_dataloader in val_dataloaders] if val_dataloaders else None
    
    train(
        accelerator=accelerator,
        placeholder_token_ids=[placeholder_token_id],
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        diffuser=diffuser,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        )
