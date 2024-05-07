import argparse
import datetime
import os
import sys
import time
import types
import warnings
import hashlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusion.utils.dist_utils import flush
from diffusion.utils.logger_ac import get_logger


logger = get_logger(__name__)


def get_text_encoding_pipeline(pipeline_load_from, device):
    logger.info(f"Loading text encoder and tokenizer from {pipeline_load_from} ...")
    tokenizer = T5Tokenizer.from_pretrained(pipeline_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        pipeline_load_from, 
        subfolder="text_encoder", 
        torch_dtype=torch.float16
    ).to(device)
    return tokenizer, text_encoder

def get_path_for_encoded_prompt(prompt, max_token_length):
    beginning = prompt[:20]
    hash_object = hashlib.sha256(prompt.encode())
    hex_dig = hash_object.hexdigest()
    tmp_dir = 'output/tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return os.path.join(tmp_dir, f'{beginning}_{hex_dig}_{max_token_length}.pth')

def encode_prompts(prompts, pipeline_load_from, device, batch_size, max_token_length):
    """
    use T5 from pipeline_load_from to encode prompts. saves to path. skips found saved encoded prompts.
    """
    need_encoding = []
    for prompt in prompts:
        path = get_path_for_encoded_prompt(
            prompt=prompt, 
            max_token_length=max_token_length,
            )
        if not os.path.exists(path):
            need_encoding.append(prompt)
    logger.info(f'encode_prompts found {len(prompts) - len(need_encoding)} saved encoded prompts. encoding {len(need_encoding)} prompts...')
    if not need_encoding:
        logger.info(f'encode_prompts: no prompts need encoding. found {len(prompts)} saved encoded prompts')
        return
        
    logger.info(f'encode_prompts found {len(prompts) - len(need_encoding)} saved encoded prompts. encoding {len(need_encoding)} prompts...')
    tokenizer, text_encoder = get_text_encoding_pipeline(
        pipeline_load_from=pipeline_load_from,
        device=device,
    )

    batches = [need_encoding[i:i + batch_size] for i in range(0, len(need_encoding), batch_size)]

    with torch.no_grad():
        for i in tqdm(range(len(batches)), desc='text encoding batches'):
            batch_prompts = batches[i]
            batch_txt_tokens = tokenizer(
                    batch_prompts, max_length=max_token_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(device)
            batch_caption_emb = text_encoder(batch_txt_tokens.input_ids, attention_mask=batch_txt_tokens.attention_mask)[0]
            
            # Assuming you are modifying to save each prompt's data separately
            for j, prompt in enumerate(batch_prompts):
                # Extract embeddings and attention mask for the j-th item in the batch
                prompt_embeds = batch_caption_emb[j, :, :]
                prompt_attention_mask = batch_txt_tokens.attention_mask[j, :]
                # Generate a unique path for each prompt
                save_path = get_path_for_encoded_prompt(
                    prompt=prompt, 
                    max_token_length=max_token_length
                    )
                torch.save({
                    'prompt_embeds': prompt_embeds,
                    'prompt_attention_mask': prompt_attention_mask,
                }, save_path)
    del tokenizer
    del text_encoder
    flush()
    logger.info(f'encode_prompts finished. encoded and saved {len(need_encoding)} prompts')
