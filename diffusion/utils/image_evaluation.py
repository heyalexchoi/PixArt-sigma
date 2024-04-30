import torch
from diffusers import ConsistencyDecoderVAE, DPMSolverMultistepScheduler, Transformer2DModel, AutoencoderKL
from scripts.diffusers_patches import pixart_sigma_init_patched_inputs, PixArtSigmaPipeline
from diffusion.utils.logger_ac import get_logger
from diffusion.utils.text_embeddings import get_path_for_encoded_prompt

logger = get_logger(__name__)

def get_image_gen_pipeline(
        pipeline_load_from,
        torch_dtype,
        device,
        transformer,
    ):
    """Get pipeline with image generation components, without text encoding. Optionally load a passed in transformer"""
    logger.info(f"Loading image gen pipeline {pipeline_load_from} to device: {device} and dtype {torch_dtype}...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        pipeline_load_from,
        transformer=transformer,
        tokenizer=None,
        text_encoder=None,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)
    
    return pipe

@torch.inference_mode()
def generate_images(
        pipeline,
        prompt_embeds,
        prompt_attention_mask,
        batch_size,
        num_inference_steps,
        width,
        height,
        device,
        max_token_length,
        seed=0,
        guidance_scale=4.5,
    ):
    """
    batch generates images from caption embeddings with batch dim
    assumes null embeddings have already been generated and saved
    """
    logger.info(f"Generating {len(prompt_embeds)} images...")    
    logger.info(f"width {width} height {height}")    
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []

    null_embed_path = get_path_for_encoded_prompt(
        prompt='',
        max_token_length=max_token_length,
        )
    null_embed = torch.load(null_embed_path)

    # Generate images in batches
    for i in range(0, len(prompt_embeds), batch_size):
        batch_prompt_embeds = prompt_embeds[i:i+batch_size].to(device)
        batch_prompt_attention_mask = prompt_attention_mask[i:i+batch_size].to(device)
        # duplicate null embeds to match batch size
        batch_size = batch_prompt_embeds.size(0)  # Get the batch size from batch_prompt_embeds
        negative_prompt_embeds = null_embed['prompt_embeds'].repeat(batch_size, 1, 1)
        negative_prompt_attention_mask = null_embed['prompt_attention_mask'].repeat(batch_size, 1)

        batch_images = pipeline(
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=batch_prompt_embeds,
            prompt_attention_mask=batch_prompt_attention_mask,
            negative_prompt=None,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        ).images
        
        images.extend(batch_images)

    return images