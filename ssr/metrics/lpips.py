import lpips
import torch
import logging

from basicsr.utils import get_root_logger
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import get_root_logger

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, lpips_model, **kwargs):
    device = torch.device('cuda')
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
    global lpipsModel
    try:
        lpipsModel
    except NameError:
        if lpips_model == 'alexnet':
            lpips_loss_fn = lpips.LPIPS(net='alex').to(device) # best forward scores
        elif lpips_model == 'vgg':
            lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

        lpipsModel = lpips_loss_fn

    tensor1 = torch.as_tensor(img).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device).float()/255

    lpips_loss = lpipsModel(tensor1, tensor2).detach().item()
    logger.info(f"LIPS: {lpips_loss}")
    return lpips_loss
