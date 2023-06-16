import hydra 

from torch.nn.parallel import DistributedDataParallel as DDP

# from .byol import BYOLLearner
# from .vicreg import VICRegLearner
# from .behavior_cloning import ImageTactileBC
# from .bet import BETLearner
# from .bc_gmm import BCGMM
# from .simclr import SIMCLRLearner
# from .mocov3 import MOCOLearner

from learner.VisualBC import VisualBC
from learner.BYOLLearner import BYOLLearner
from learner.temporal_ssl import TemporalSSLLearner

# from tactile_learning.utils import *
# from tactile_learning.models import  *

from utils import *
from model import *

def init_learner(cfg, device, rank=0):
    if cfg.learner_type == 'bc':
        return init_bc(cfg, device, rank)
    elif cfg.learner_type == 'image_byol':
        return init_image_byol(cfg, device, rank)
    elif cfg.learner_type == 'temporal_ssl':
        return init_temporal_learner(cfg, device, rank)
    return None

def init_bc(cfg, device, rank):
    image_encoder = hydra.utils.instantiate(cfg.encoder.image_encoder).to(device)
    image_encoder = DDP(image_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # tactile_encoder = hydra.utils.instantiate(cfg.encoder.tactile_encoder).to(device)
    # tactile_encoder = DDP(tactile_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    last_layer = hydra.utils.instantiate(cfg.encoder.last_layer).to(device)
    last_layer = DDP(last_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optim_params = list(image_encoder.parameters()) + list(last_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    # learner = ImageTactileBC(
    #     image_encoder = image_encoder, 
    #     tactile_encoder = tactile_encoder,
    #     last_layer = last_layer,
    #     optimizer = optimizer,
    #     loss_fn = cfg.learner.loss_fn,
    #     representation_type = cfg.learner.representation_type,
    #     freeze_encoders = cfg.learner.freeze_encoders
    # )
    learner = VisualBC(
        image_encoder = image_encoder, 
        last_layer = last_layer,
        optimizer = optimizer,
        loss_fn = cfg.learner.loss_fn,
        representation_type = cfg.learner.representation_type,
        freeze_encoders = cfg.learner.freeze_encoders
    )
    learner.to(device) 
    
    return learner




def init_image_byol(cfg, device, rank):
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_vision_augmentations(
        img_means = VISION_IMAGE_MEANS,
        img_stds = VISION_IMAGE_STDS,
    )

    ## model: nothing to be done about it
    byol = BYOL(
        net = encoder,
        image_size = cfg.vision_image_size,
        augment_fn = augment_fn
    ).to(device)

    #encoder is only set to be distributed here to avoid BYOL to set any torch on differenty GPUs
    # encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'image'
    )

    learner.to(device)

    return learner



def init_temporal_learner(cfg, device, rank):
    encoder = hydra.utils.instantiate(cfg.encoder.encoder).to(device)
    if cfg.distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    linear_layer = hydra.utils.instantiate(cfg.encoder.linear_layer).to(device)
    if cfg.distributed:
        linear_layer = DDP(linear_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(encoder.parameters()) + list(linear_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=optim_params)

    learner = TemporalSSLLearner(
        optimizer = optimizer,
        repr_loss_fn = cfg.learner.repr_loss_fn,
        joint_diff_loss_fn = cfg.learner.joint_diff_loss_fn,
        encoder = encoder,
        linear_layer = linear_layer,
        joint_diff_scale_factor = cfg.learner.joint_diff_scale_factor,
        total_loss_type = cfg.learner.total_loss_type
    )
    learner.to(device)

    return learner