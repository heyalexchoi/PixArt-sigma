import os
import numpy as np
import torch
import random
from torchvision.datasets.folder import default_loader
from diffusion.data.datasets.InternalData import InternalData, InternalDataSigma
from diffusion.data.builder import get_data_path, DATASETS
from diffusion.utils.logger import get_root_logger
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from diffusion.data.datasets.utils import *
from diffusion.data.datasets.utils_ac import get_vae_feature_path, get_t5_feature_path, get_vae_signature
from diffusion.utils.logger_ac import get_logger

logger = get_logger(__name__)

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

# simplified, cleaned up
# assumes pre-extracted features, no handling of raw text or images
# no 'real prompt ratio', etc
# uses vae and t5 feat path from utils
@DATASETS.register_module()
class InternalDataMSSigmaAC(InternalDataSigma):
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=True,
                 load_t5_feat=True,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 mask_type='null',
                 load_mask_index=False,
                 vae_save_root=None,
                 t5_save_dir=None,
                 vae_type='sdxl',
                 max_token_length=300,
                 config=None,
                 conditional_dropout=0.0,
                 null_embed_path=None,
                 **kwargs):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.load_t5_feat = load_t5_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.mask_type = mask_type
        self.max_token_length = max_token_length
        self.base_size = int(kwargs['aspect_ratio_type'].split('_')[-1])
        self.aspect_ratio = eval(kwargs.pop('aspect_ratio_type'))       # base aspect ratio
        self.meta_data_clean = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.ratio_index = {}
        self.ratio_nums = {}
        self.weight_dtype = torch.float16
        self.interpolate_model = InterpolationMode.BICUBIC
        self.conditional_dropout = conditional_dropout
        self.null_embed_path = null_embed_path
        if self.aspect_ratio in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]:
            self.interpolate_model = InterpolationMode.LANCZOS
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler
        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.info(f"T5 max token length: {self.max_token_length}")

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, json_file))
            logger.info(f"{json_file} data volume: {len(meta_data)}")
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.5]
            self.meta_data_clean.extend(meta_data_clean)

            self.txt_feat_samples.extend([
                get_t5_feature_path(
                    t5_save_dir=t5_save_dir,
                    image_path=item['path'],
                    max_token_length=max_token_length,
                    relative_root_dir=self.root,
                ) for item in meta_data_clean
            ])
            self.vae_feat_samples.extend([
                get_vae_feature_path(
                    resolution=resolution,
                    vae_type=vae_type,
                    is_multiscale=True,
                    vae_save_root=vae_save_root, 
                    image_path=item['path'],
                    relative_root_dir=self.root,
                ) for item in meta_data_clean
            ])

        # Set loader and extensions
        self.transform = None
        self.loader = self.vae_feat_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

    def get_item(self, index):
        return self.meta_data_clean[index]
    
    def get_item_dimensions(self, index):
        item = self.get_item(index)
        ori_h, ori_w = item['height'], item['width']
        return ori_h, ori_w
    
    def get_closest_ratio(self, height, width):
        closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratio)
        return closest_size, closest_ratio

    def getdata(self, index):
        npz_path = self.txt_feat_samples[index]
        npy_path = self.vae_feat_samples[index]
        data_info = {}
        ori_h, ori_w = self.get_item_dimensions(index)

        # Calculate the closest aspect ratio and resize & crop image[w, h]
        _, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)

        img = self.loader(npy_path)
        if index not in self.ratio_index[closest_ratio]:
            self.ratio_index[closest_ratio].append(index)
        h, w = (img.shape[1], img.shape[2])
        assert h, w == (ori_h//8, ori_w//8)

        data_info['img_hw'] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info['aspect_ratio'] = closest_ratio
        data_info["mask_type"] = self.mask_type

        attention_mask = torch.ones(1, 1, self.max_token_length)
        
        if (self.conditional_dropout > 0 
            and self.null_embed_path is not None
            and random.random() < self.conditional_dropout):
            logger.info(f'loading null embedding for cond dropout {self.conditional_dropout}')
            # load null embedding
            txt_info = torch.load(self.null_embed_path)
            txt_fea = txt_info['prompt_embeds'][None]
            attention_mask = txt_info['prompt_attention_mask'][None]
            logger.info(f'txt_info: {txt_info.keys()}\n{txt_info["prompt_embeds"].shape}\n{txt_info["prompt_attention_mask"].shape}')
        else:
            logger.info('loading t5 embedding')
            txt_info = np.load(npz_path)
            # add batch dimension to get to shape torch.Size([1, 300, 4096])
            txt_fea = torch.from_numpy(txt_info['caption_feature'])[None]
            if 'attention_mask' in txt_info.keys():
                attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        # pad to max_token_length
        if txt_fea.shape[1] != self.max_token_length:
            logger.warn(f'dataloader txt_fea.shape[1] {txt_fea.shape[1]} does not match self.max_token_length: {self.max_token_length}')
            txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_token_length-txt_fea.shape[1], 1)], dim=1).to(self.weight_dtype)
            attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_token_length-attention_mask.shape[-1])], dim=-1)

        return img, txt_fea, attention_mask.to(torch.int16), data_info

    def __len__(self):
        return len(self.meta_data_clean)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                logger.exception(f'dataloader exception __getitem__: {e}')
                ori_h, ori_w = self.get_item_dimensions(idx)
                _, closest_ratio = self.get_closest_ratio(ori_h, ori_w)
                idx = random.choice(self.ratio_index[closest_ratio])
        raise RuntimeError('InternalDataMSSigmaAC.__getitem__ failed too many times')

