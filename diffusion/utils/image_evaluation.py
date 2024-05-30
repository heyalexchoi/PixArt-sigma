import torch
from diffusers import PixArtSigmaPipeline
from diffusion.utils.logger_ac import get_logger
from diffusion.utils.text_embeddings import get_path_for_encoded_prompt
from torch.utils.data import Dataset, DataLoader
from diffusion.utils.dist_utils import flush

logger = get_logger(__name__)

class PromptDataset(Dataset):
    def __init__(self, prompt_embeds, prompt_attention_mask, negative_prompt_embeds=None, negative_prompt_attention_mask=None):
        self.prompt_embeds = prompt_embeds
        self.prompt_attention_mask = prompt_attention_mask
        self.negative_prompt_embeds = negative_prompt_embeds
        self.negative_prompt_attention_mask = negative_prompt_attention_mask

    def __len__(self):
        return len(self.prompt_embeds)

    def __getitem__(self, idx):
        item = {
            'prompt_embeds': self.prompt_embeds[idx],
            'prompt_attention_mask': self.prompt_attention_mask[idx]
        }
        if self.negative_prompt_embeds is not None and self.negative_prompt_attention_mask is not None:
            item['negative_prompt_embeds'] = self.negative_prompt_embeds[idx]
            item['negative_prompt_attention_mask'] = self.negative_prompt_attention_mask[idx]
        return item

def get_image_gen_pipeline(
        pipeline_load_from,
        torch_dtype,
        device,
        transformer,
        text_encoder=None,
    ):
    """Get pipeline with image generation components, without text encoding. Optionally load a passed in transformer"""
    logger.info(f"Loading image gen pipeline {pipeline_load_from} to device: {device} and dtype {torch_dtype}...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        pipeline_load_from,
        transformer=transformer,
        tokenizer=None,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)
    
    return pipe

@torch.inference_mode()
def generate_images(
        pipeline,
        accelerator,
        prompt_embeds,
        prompt_attention_mask,
        batch_size,
        num_inference_steps,
        width,
        height,
        device,
        max_token_length,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        seed=0,
        guidance_scale=4.5,
        output_type='pil',
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

    dataset = PromptDataset(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = accelerator.prepare(dataloader)
        
    # Generate images in batches
    # for i in range(0, len(prompt_embeds), batch_size):
    for index, batch in enumerate(dataloader):
        logger.info(f"Generating images for batch {index}/{len(dataloader)}")
        # batch_prompt_embeds = prompt_embeds[i:i+batch_size].to(device)
        batch_prompt_embeds = batch['prompt_embeds'].to(device)
        # batch_prompt_attention_mask = prompt_attention_mask[i:i+batch_size].to(device)
        batch_prompt_attention_mask = batch['prompt_attention_mask'].to(device)
        # duplicate null embeds to match batch size
        actual_batch_size = batch_prompt_embeds.size(0)  # Get the actual batch size from batch_prompt_embeds
        batch_negative_prompt_embeds = null_embed['prompt_embeds'].repeat(actual_batch_size, 1, 1)
        batch_negative_prompt_attention_mask = null_embed['prompt_attention_mask'].repeat(actual_batch_size, 1)
        # if negative_prompt_embeds is not None and negative_prompt_attention_mask is not None:
        if 'negative_prompt_embeds' in batch and 'negative_prompt_attention_mask' in batch:
            # batch_negative_prompt_embeds = negative_prompt_embeds[i:i+batch_size].to(device)
            # batch_negative_prompt_attention_mask = negative_prompt_attention_mask[i:i+batch_size].to(device)
            batch_negative_prompt_embeds = batch['negative_prompt_embeds'].to(device)
            batch_negative_prompt_attention_mask = batch['negative_prompt_attention_mask'].to(device)
            logger.info('using negative prompt embeds')

        batch_images = pipeline(
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=batch_prompt_embeds,
            prompt_attention_mask=batch_prompt_attention_mask,
            negative_prompt_embeds=batch_negative_prompt_embeds,
            negative_prompt_attention_mask=batch_negative_prompt_attention_mask,
            negative_prompt=None, # this has to be explicitly set to None if using negative prompt embeds
            output_type=output_type,
        ).images

        batch_images = accelerator.gather(batch_images)
        logger.info(f"Finished Generating images for batch {index}/{len(dataloader)}")
        images.extend(batch_images)

    return images