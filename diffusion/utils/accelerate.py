import torch
from diffusion.utils.dist_utils import synchronize

def wait_for_everyone():
    # possible issue with accelerator.wait_for_everyone() breaking on linux kernel < 5.5
    # https://github.com/huggingface/accelerate/issues/1929
    synchronize()

dtype_mapping = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
        'fp64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'no': torch.float32,
    }

# never successfully used this one
def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'