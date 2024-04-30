import os

# vae should include distinction between 
# - ldm vae and sdxl vae. uploaded paths already assume ldm, so maybe new dir for sdxl vae.
# - MS vs cropped. I think I have this
# - resolution. I think I have this
# t5 probably will work same. just length is varied.

def get_vae_feature_path(
        vae_save_root, 
        image_path, 
        vae_type,
        resolution,
        is_multiscale,
        relative_root_dir=None):
    signature = get_vae_signature(
        resolution=resolution, 
        is_multiscale=is_multiscale, 
        vae_type=vae_type
        )
    root_dir = os.path.join(vae_save_root, signature)

    return get_feature_path(
        feature_dir=root_dir, 
        image_path=image_path,
        relative_root_dir=relative_root_dir,
        extension='.npy')

def get_t5_feature_path(t5_save_dir, image_path, max_token_length=300, relative_root_dir=None):
    signature = f"max-{max_token_length}"
    root_dir = os.path.join(t5_save_dir, signature)
    return get_feature_path(
                feature_dir=root_dir, 
                image_path=image_path,
                relative_root_dir=relative_root_dir,
                extension='.npz',
            )

def get_feature_path(feature_dir, image_path, extension, relative_root_dir):
    """
    Returns full feature path in feature_dir, first creating safe filename from image_path, relative to relative_root_dir
    """
    if relative_root_dir:
        absolute_image_path = os.path.abspath(image_path)
        absolute_root_dir = os.path.abspath(relative_root_dir)
        # Check if the root dir is part of the image path
        common_path = os.path.commonpath([absolute_image_path, absolute_root_dir])
        if common_path == absolute_root_dir:
            # Make the image path relative to the root dir
            image_path = os.path.relpath(absolute_image_path, absolute_root_dir)
    
    safe_name = image_path.replace('/', '---').replace('\\', '---')
    safe_name_no_ext = safe_name.rsplit('.', 1)[0]  # Remove the original extension
    if not extension.startswith('.'):
        extension = '.' + extension
    # Return the transformed path as a safe filename with the new extension
    return os.path.join(feature_dir, safe_name_no_ext + extension)

# should pass in vae type 'sdxl' or 'ldm'
# previous extractions used 'ldm' and were only type, so the vae type part of the signature is not included
def get_vae_signature(resolution, is_multiscale, vae_type):
    assert resolution in [256, 512, 1024, 2048]
    first_part = 'multiscale' if is_multiscale else 'cropped'
    signature = f"{first_part}-{resolution}"
    if vae_type == 'sdxl':
        return f"sdxl-{signature}"
    if vae_type == 'ldm':
        return signature
    else:
        raise ValueError(f"Unknown VAE type {vae_type}")

