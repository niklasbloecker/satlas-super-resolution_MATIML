import clip
import torch
import logging
import open_clip
import torch.nn.functional as F
import logging
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_clipscore(img, img2, clip_model, **kwargs):
    device = torch.device('cuda')
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

    if clip_model == 'clip-ViT-B/16':
        model, _ = clip.load("ViT-B/16", device=device)
        img_size = (224,224)
        logger.info(f"{'clip-ViT-B/16'} loaded with img size {img_size}")
    elif clip_model == 'clipa-ViT-bigG-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
        model = model.to(device)
        img_size = (336,336)
        logger.info(f"{'clipa-ViT-bigG-14'} loaded with img size {img_size}")
    elif clip_model == 'siglip-ViT-SO400M-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
        model = model.to(device)
        img_size = (384,384)
        logger.info(f"{'siglip-ViT-SO400M-14'} loaded with img size {img_size}")
    else:
        print(clip_model, " is not supported for CLIPScore.")

    tensor1 = torch.as_tensor(img).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device).float()/255

    tensor1 = F.interpolate(tensor1, img_size)
    tensor2 = F.interpolate(tensor2, img_size)

    feats1 = model.encode_image(tensor1)
    feats2 = model.encode_image(tensor2)

    clip_score = F.cosine_similarity(feats1, feats2).detach().item()
    logger.info(f"CLIP Score: {clip_score}")
    return clip_score
