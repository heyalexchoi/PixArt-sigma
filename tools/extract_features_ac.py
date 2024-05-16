import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue
import concurrent.futures
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.folder import default_loader

from diffusion.utils.logger_ac import get_logger
from diffusers.models import AutoencoderKL
from transformers import T5Tokenizer, T5EncoderModel
from diffusion.data.datasets.InternalData import InternalData
from diffusion.utils.misc import SimpleTimer
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.data.builder import DATASETS
from diffusion.data import ASPECT_RATIO_512, ASPECT_RATIO_1024, ASPECT_RATIO_256
from diffusion.data.datasets.utils_ac import get_vae_feature_path, get_t5_feature_path, get_vae_signature
from diffusion.utils.dist_utils import flush

from concurrent.futures import ProcessPoolExecutor, wait, as_completed


logger = get_logger(__name__)

sdxl_vae_path = 'madebyollin/sdxl-vae-fp16-fix'

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

# VAE feature extraction
@DATASETS.register_module()
class DatasetMS(InternalData):
    def __init__(self, root, multi_scale, image_list_json=None,
                 aspect_ratio_type=None, start_index=0,
                 end_index=100000000, **kwargs):
        if image_list_json is None:
            image_list_json = ['data_info.json']
        assert os.path.isabs(root), 'root must be a absolute path'
        self.root = root
        self.meta_data_clean = []
        self.img_samples = []
        self.aspect_ratio = aspect_ratio_type
        assert self.aspect_ratio in [ASPECT_RATIO_1024, ASPECT_RATIO_512, ASPECT_RATIO_256]
        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        vae_already_processed = []
        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, 'partition', json_file))
            logger.info(f'json_file: {json_file} has {len(meta_data)} meta_data')
            # filter by ratio and already extracted VAE features
            for item in meta_data:
                if item['ratio'] <= 4:
                    sample_path = os.path.join(self.root, item['path'])
                    # this dataset seems to be for multiscale vae extraction only
                    output_file_path = get_vae_feature_path(
                        resolution=image_resize,
                        vae_type=vae_type,
                        is_multiscale=multi_scale,
                        vae_save_root=vae_save_root, 
                        image_path=sample_path,
                        relative_root_dir=self.root,
                        )
                    if not os.path.exists(output_file_path):
                        self.meta_data_clean.append(item)
                        self.img_samples.append(sample_path)
                    else:
                        vae_already_processed.append(sample_path)

        logger.info(f"VAE processing skipping {len(vae_already_processed)} images already processed")

        self.img_samples = self.img_samples[start_index: end_index]
        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

        # Set loader and extensions
        self.loader = default_loader

    def __getitem__(self, idx):
        data_info = {}
        try:
            img_path = self.img_samples[idx]
            img = self.loader(img_path)
            # Calculate closest aspect ratio and resize & crop image[w, h]
            if isinstance(img, Image.Image):
                h, w = (img.size[1], img.size[0])
                assert h, w == (self.meta_data_clean[idx]['height'], self.meta_data_clean[idx]['width'])
                closest_size, closest_ratio = get_closest_ratio(h, w, self.aspect_ratio)
                closest_size = list(map(lambda x: int(x), closest_size))
                # TODO: non-multiscale transformation: crop / resize to square and skip closest size part
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB')),
                    # TODO maybe: use single dimension to resize preserving aspect ratio. current 2 dimension resize can warp image.
                    T.Resize(closest_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                    T.CenterCrop(closest_size),
                    T.ToTensor(),
                    T.Normalize([.5], [.5]),
                ])
                img = transform(img)
                data_info['img_hw'] = torch.tensor([h, w], dtype=torch.float32)
                data_info['aspect_ratio'] = closest_ratio
            # change the path according to your data structure
            return img, self.img_samples[idx]
        except Exception as e:
            logger.exception(f"Error details: {str(e)}")
        
    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

def save_results(results, paths, vae_save_root):
    # save to npy
    new_paths = []
    os.umask(0o000)  # file permission: 666; dir permission: 777
    for res, p in zip(results, paths):
        output_path = get_vae_feature_path(
            vae_save_root=vae_save_root,
            is_multiscale=multi_scale,
            resolution=image_resize,
            vae_type=vae_type,
            image_path=p,
            relative_root_dir=dataset_root,
            )
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        dirname_base = os.path.basename(dirname)
        filename = os.path.basename(output_path)
        new_paths.append(os.path.join(dirname_base, filename))
        np.save(output_path, res)
    # save paths
    signature = get_vae_signature(
        resolution=image_resize,
        is_multiscale=multi_scale,
        vae_type=vae_type,
    )
    with open(os.path.join(vae_save_root, f"VAE-{signature}.txt"), 'a') as f:
        f.write('\n'.join(new_paths) + '\n')

def inference(vae, dataloader, vae_save_root):
    timer = SimpleTimer(len(dataloader), log_interval=1, desc="VAE-Inference")

    for batch in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                posterior = vae.encode(batch[0]).latent_dist
                results = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        path = batch[1]
        save_results(results=results, paths=path, vae_save_root=vae_save_root)
        timer.log()


def extract_img_vae_multiscale(batch_size, device, num_workers):
    assert image_resize in [256, 512, 1024]
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(vae_save_root, exist_ok=True)
    if device == 'cuda':
        accelerator = Accelerator(mixed_precision='fp16')
    else:
        accelerator = Accelerator()
    vae = AutoencoderKL.from_pretrained(f'{args.vae_models_dir}', torch_dtype=torch.float16).to(device)

    aspect_ratio_type = {
        256: ASPECT_RATIO_256,
        512: ASPECT_RATIO_512,
        1024: ASPECT_RATIO_1024
    }[image_resize]
    dataset = DatasetMS(dataset_root, image_list_json=[json_file], transform=None, sample_subset=None,
                        aspect_ratio_type=aspect_ratio_type, start_index=start_index, end_index=end_index,
                        multi_scale=multi_scale,
                        )

    # create AspectRatioBatchSampler
    sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset, batch_size=batch_size, aspect_ratios=dataset.aspect_ratio, ratio_nums=dataset.ratio_nums)

    # create DataLoader
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    dataloader = accelerator.prepare(dataloader, )

    inference(vae, dataloader, vae_save_root=vae_save_root)
    accelerator.wait_for_everyone()
    logger.info('finished extract_img_vae_multiscale')

# T5 feature extraction
def extract_caption_t5_batch(batch, tokenizer, text_encoder, t5_save_dir, t5_max_token_length, dataset_root):
    logger.info(f'extract_caption_t5_batch start') #
    with torch.no_grad():
        captions = [item['prompt'].strip() for item in batch]
        output_paths = [get_t5_feature_path(
            t5_save_dir=t5_save_dir, 
            image_path=item['path'],
            relative_root_dir=dataset_root,
            max_token_length=t5_max_token_length,
        ) for item in batch]

        for output_path in output_paths:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # caption_embs, emb_masks = t5.get_text_embeddings(captions)
        caption_tokens = tokenizer(
            captions,
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
            ).to(device)
        caption_embeddings = text_encoder(
                        caption_tokens.input_ids, 
                        attention_mask=caption_tokens.attention_mask
                        )[0]
        save_futures = []
        for i, output_path in enumerate(output_paths):
            emb_dict = {
                'caption_feature': caption_embeddings[i].to(torch.float16).cpu().data.numpy(),
                'attention_mask': caption_tokens.attention_mask[i].to(torch.float16).cpu().data.numpy(),
            }
            save_future = async_save_embedding(
                output_path=output_path,
                emb_dict=emb_dict,
            )
            save_futures.append(save_future)
        logger.info(f"Completed T5 batch of length {len(batch)}")
        return save_futures

def extract_caption_t5(
        t5_batch_size, 
        device, 
        t5_save_dir, 
        json_path, 
        t5_max_token_length, 
        dataset_root,
        ):
    os.makedirs(t5_save_dir, exist_ok=True)
    train_data_json = json.load(open(json_path, 'r'))
    # Assuming args.start_index and args.end_index are defined elsewhere
    train_data = train_data_json[start_index:end_index]
    logger.info('Calculating completed paths...') #
    completed_paths = set([item['path'] for item in train_data if os.path.exists(get_t5_feature_path(
        t5_save_dir=t5_save_dir, 
        image_path=item['path'],
        relative_root_dir=dataset_root,
        max_token_length=t5_max_token_length,
    ))])
    train_data = [item for item in train_data if item['path'] not in completed_paths]
    logger.info(f"Skipping T5 extraction for {len(completed_paths)} items with existing .npz files.")
    logger.info('Loading T5Embedder...')
    tokenizer = T5Tokenizer.from_pretrained(args.t5_models_dir, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.t5_models_dir, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    batch_size = t5_batch_size
    batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
    logger.info(f'Processing {len(batches)} batches of batch_size {batch_size}')

    all_save_futures = []
    for i in tqdm(range(len(batches)), desc="Processing Batches"):
        batch = batches[i]
        batch_save_futures = extract_caption_t5_batch(
            batch=batch, 
            text_encoder=text_encoder,
            tokenizer=tokenizer, 
            t5_save_dir=t5_save_dir, 
            t5_max_token_length=t5_max_token_length, 
            dataset_root=dataset_root,
            )
        all_save_futures.extend(batch_save_futures)            

    logger.info('finished extract_caption_t5. cleaning up...')
    del t5
    flush()
    return all_save_futures

def save_emb_dict(output_path, emb_dict):
    np.savez_compressed(output_path, **emb_dict)

def async_save_embedding(output_path, emb_dict):
    future = executor.submit(save_emb_dict, output_path, emb_dict)
    return future

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--multi_scale", action='store_true', default=False, help="multi-scale feature extraction")
    parser.add_argument("--img_size", default=512, type=int, choices=[256, 512, 1024], help="image scale for multi-scale feature extraction")
    parser.add_argument('--vae_batch_size', default=1, type=int)
    parser.add_argument('--t5_batch_size', default=1, type=int)
    parser.add_argument('--t5_max_token_length', default=120, type=int)
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=1000000, type=int)
    parser.add_argument('--t5_save_root', default='data/data_toy/caption_feature_wmask', type=str)
    parser.add_argument('--vae_save_root', default='data/data_toy/img_vae_features', type=str)
    parser.add_argument('--dataset_root', default='data/data_toy', type=str)
    # parser.add_argument('--pretrained_models_dir', default='output/pretrained_models', type=str)
    parser.add_argument('--max_workers', default=8, type=int, help="Maximum workers")
    parser.add_argument(
        '--t5_models_dir', default='PixArt-alpha/PixArt-XL-2-1024-MS', type=str
    )
    parser.add_argument(
        '--vae_models_dir', default=sdxl_vae_path, type=str
    )
    parser.add_argument('--skip_t5', action='store_true', default=False, help="skip t5 feature extraction")
    parser.add_argument('--skip_vae', action='store_true', default=False, help="skip vae feature extraction")

    ### for multi-scale(ms) vae feauture extraction
    parser.add_argument('--json_file', type=str)

    # override device
    parser.add_argument('--device', default='cuda', type=str, help="device to use for feature extraction")
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    device = 'cuda'
    if args.device == 'mps':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        else:
            logger.warning("MPS is not available. Using cuda instead.")
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using cpu instead.")
        device = 'cpu'

    image_resize = args.img_size
    multi_scale = True
    vae_save_root = os.path.abspath(args.vae_save_root)
    t5_save_dir = args.t5_save_root
    vae_models_dir = args.vae_models_dir
    if not vae_models_dir == sdxl_vae_path:
        raise ValueError(f"Unhandled VAE model: {vae_models_dir}")
    vae_type = 'sdxl'
    
    json_file = args.json_file
    json_path = json_file # pretty sure this is just duplicate. can clean this up later
    t5_max_token_length = args.t5_max_token_length
    dataset_root = args.dataset_root
    vae_batch_size = args.vae_batch_size
    t5_batch_size = args.t5_batch_size
    max_workers = args.max_workers

    start_index = args.start_index
    end_index = args.end_index

    executor = ProcessPoolExecutor(max_workers=max_workers)
    # disable warning from tokenizers. 
    # parallelization here should all be downstream of tokenizers
    # make sure not parallelizing any tokenization in my code.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    t5_save_futures = None
    if not args.skip_t5:
        # prepare extracted caption t5 features for training
        logger.info(f"Extracting T5 features for {json_path}\nMax token length: {t5_max_token_length}\
                    \nDevice: {device}\nBatch size: {t5_batch_size}\nMax Workers: {max_workers}\
                    \nSave to: {t5_save_dir}")
        
        t5_save_futures = extract_caption_t5(
            t5_batch_size=t5_batch_size,
            device=device,
            t5_save_dir=t5_save_dir,
            json_path=json_path,
            t5_max_token_length=t5_max_token_length,
            dataset_root=dataset_root,
        )

    if not args.skip_vae:
        # prepare extracted image vae features for training
        logger.info(f"Extracting VAE features for {json_path}\nmulti_scale: {multi_scale}\
                    \nimage_resize: {image_resize}\nDevice: {device}\nBatch Size: {vae_batch_size}\
                    \nvae_models_dir: {vae_models_dir} vae_type: {vae_type} Save to: {vae_save_root}")
        if not multi_scale:
            # basically seemed like the two did the same thing except one code path was shittier
            # and the non-multi-scale cropped to square instead of looking for nearest aspect ratio
            raise ValueError("Single scale feature extraction is not supported currently.")
            
        # recommend bs = 1 for AspectRatioBatchSampler
        # not sure why bs = 1 is recommended. bigger batches are used in training. try higher.
        extract_img_vae_multiscale(
            batch_size=vae_batch_size, 
            device=device,
            num_workers=max_workers,
            )
    
    if t5_save_futures:
        # Use tqdm with as_completed to show progress
        for future in tqdm(as_completed(t5_save_futures), total=len(t5_save_futures), desc="T5 Save Embeds"):
            try:
                result = future.result()
                # Optionally process or log result here
            except Exception as exc:
                print(f"T5 Save Embedding Task generated an exception: {exc}")

        # wait for all_save_futures 
        done, not_done = wait(t5_save_futures)  # This will block until all futures are done