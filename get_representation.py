import glob
import hydra 
from omegaconf import DictConfig

from datasets.preprocess import *
from model.utils import * 
from utils.augmentation import *
from utils.data import *

from holobot.samplers.allegro import AllegroSampler 


@hydra.main(version_base=None, config_path='configs', config_name='get_representation')
def main(cfg : DictConfig) -> None:
    image_out_dir = cfg.data_out_dir
    data_path = cfg.data_path
    device = torch.device('cuda:0') 
    demos_to_use = cfg.demos_to_use
    representation_types = cfg.representation_types
    save_path = cfg.save_path

    image_cfg, image_encoder, image_transform = init_encoder_info(device, image_out_dir)
    inv_image_transform = get_inverse_image_norm()
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))

    print('Getting all representations')
    repr_dim = 0
    if 'allegro' in representation_types:  repr_dim += ALLEGRO_EE_REPR_SIZE
    if 'kinova' in representation_types: repr_dim += KINOVA_JOINT_NUM
    if 'torque' in representation_types: repr_dim += ALLEGRO_JOINT_NUM # There are 16 joint values
    if 'image' in representation_types: repr_dim += image_cfg.encoder.out_dim
    print(roots)
    all_representations = np.zeros((
                          0, repr_dim
                          ))
    demo_img = []
        
    ## Build the sampler and sample all the timestamps:
    for demo_id, root in enumerate(roots):
        if demo_id in [0,1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 17, 18, 19]:
        # if demo_id in [0,1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]:
            continue
        print("demo id:{}".format(demo_id))
        sampler = AllegroSampler(root, [0], 'rgb', 0.01)
        sampler.sample_data()
        ##store the image and state information: 
        for index in range(len(sampler.sampled_robot_states)):
            ## states
            representation = sampler.sampled_robot_states[index]
            ## image and preproccessing
            image = _load_dataset_image(data_path, demo_id, sampler.sampled_rgb_frame_idxs[0][index], image_transform).to(device)
            image = image_encoder(image.unsqueeze(dim=0)) # Add a dimension to the first axis so that it could be considered as a batch
            image = image.detach().cpu().numpy().squeeze()
            # representation += image
            representation = np.concatenate([image, representation], axis=0)
            all_representations=np.vstack((all_representations, representation))
            #Now we have to keep the demo_ids and the image_ids as well
            demo_img.append([demo_id, sampler.sampled_rgb_frame_idxs[0][index]])
        print("---------------------------------------------------------num of frames sampled {}, demo: {}".format(len(sampler.sampled_rgb_frame_idxs[0]),demo_id))
    
    demo_img = np.array(demo_img)
    os.makedirs(save_path, exist_ok=True)
    #dump all representations to an npy file
    np.save(os.path.join(save_path, 'representation'), all_representations)
    np.save(os.path.join(save_path, 'demo_image_ids'), demo_img)
    print("All representation saved!")
    return

def init_encoder_info(device, image_out_dir):    
    cfg = OmegaConf.load(os.path.join(image_out_dir, '.hydra/config.yaml'))
    model_path = os.path.join(image_out_dir, 'saved_models/byol_encoder_best.pt')
    encoder = load_model(cfg, device, model_path)
    encoder.eval() 
        
    transform = T.Compose([
        T.Resize((480,680)),
        T.Lambda(crop_transform),
        T.Resize(480),
        T.ToTensor(),
        T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
    ])

    return cfg, encoder, transform

def _load_dataset_image(data_path, demo_id, image_id, image_transform):
    dset_img = load_dataset_image(data_path, demo_id, image_id, 0)
    img = image_transform(dset_img)
    return torch.FloatTensor(img) 

if __name__ == '__main__':
    main()