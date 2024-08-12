import clip
import torch
import logging
import open_clip
import logging

from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from basicsr.utils import get_root_logger
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_remote_clipscore(img, img2, clip_model, **kwargs):
    global remoteClipModel
    device = torch.device('cuda')
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

    try: remoteClipModel
    except NameError:
        if clip_model == 'RN50':
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{clip_model}.pt", cache_dir='checkpoints')
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model)
            #img_size = (224,224)
            logger.info(f'{clip_model} is downloaded to {checkpoint_path}.')
        elif clip_model == 'ViT-B-32':
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{clip_model}.pt", cache_dir='checkpoints')
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model)
            model = model.to(device)
            #img_size = (224,224)
            logger.info(f'{clip_model} is downloaded to {checkpoint_path}.')
        elif clip_model == 'ViT-L-14':
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{clip_model}.pt", cache_dir='checkpoints')
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model)
            model = model.to(device)
            #img_size = (224,224)
            logger.info(f'{clip_model} is downloaded to {checkpoint_path}.')
        else:
            print(clip_model, " is not supported for CLIPScore.")

        model = model.cuda().eval()
        remoteClipModel = model
    
    img_size = (224,224)
    tensor1 = torch.as_tensor(img).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device).float()/255

    tensor1 = F.interpolate(tensor1, img_size)
    tensor2 = F.interpolate(tensor2, img_size)

    feats1 = remoteClipModel.encode_image(tensor1)
    feats2 = remoteClipModel.encode_image(tensor2)

    clip_score = F.cosine_similarity(feats1, feats2).detach().item()
    logger.info(f"Remote CLIP Score: {clip_score}")
    return clip_score