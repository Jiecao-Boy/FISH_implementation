import glob
import h5py
import numpy as np
import os
import pickle
import torch

from torchvision.datasets.folder import default_loader as loader
from tqdm import tqdm

# Method to load all the data from given roots and return arrays for it
#root should be home/Desktop/MetaWorld/demos/hammer-v2
def load_data(root, demos_to_use=[], duration=120): # If the total length is equal to 2 hrs - it means we want the whole data
    roots = sorted(root)


    # tactile_indices = [] 
    # allegro_indices = []
    # allegro_action_indices = [] 
    # kinova_indices = []
    
    # tactile_values = {}
    # allegro_tip_positions = {} 
    allegro_joint_positions = {}
    allegro_joint_torques = {}
    allegro_actions = {}
    # kinova_states = {}
    length=[]
    demo={}

    ## hace to change this, there is only one pkl file for each demo
    for demo_id,root in enumerate(roots): 
        demo_num = int(root.split('/')[-1].split('_')[-1])
        if (len(demos_to_use) > 0 and demo_num in demos_to_use) or (len(demos_to_use) == 0): # If it's empty then it will be ignored
    #         with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
    #             tactile_indices += pickle.load(f)
    #         with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
    #             allegro_indices += pickle.load(f)
    #         with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
    #             allegro_action_indices += pickle.load(f)
    #         with open(os.path.join(root, 'kinova_indices.pkl'), 'rb') as f:
    #             kinova_indices += pickle.load(f)
    #         with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
    #             image_indices += pickle.load(f)

            # Load the data
            # with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
            #     allegro_tip_positions[demo_id] = f['positions'][()]
              with h5py.File(os.path.join(root, 'allegro_joint_states.h5'), 'r') as f:
                  allegro_joint_positions[demo_id] = f['positions'][()]
                  allegro_joint_torques[demo_id] = f['efforts'][()]
              with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
                  allegro_actions[demo_id] = f['positions'][()] # Positions are to be learned - since this is a position control
    #         with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
    #             tactile_values[demo_id] = f['sensor_values'][()]
    #         with h5py.File(os.path.join(root, 'kinova_cartesian_states.h5'), 'r') as f:
    #             state = np.concatenate([f['positions'][()], f['orientations'][()]], axis=1)     
    #             kinova_states[demo_id] = state

              ## getting the number of images in each root
              images = glob.glob(os.path.join(root, 'cam_0_rgb_images/*.png'))
              length.append((demo_id, len(images))) 
              demo[demo_id] = demo_num        




    # for root in roots:
    #     for demo_id in range(6): #we have 6 demos
    #         demo_id += 1
    #         file_name = f'expert_demos_{demo_id}.pkl'
    #         with open(os.path.join(root, file_name), 'rb') as f:
    #             output = pickle.load(f)
    #             image+=(output[0])
    #             observations+=(output[1])
    #             actions+=(output[2])
    #             #get the length of each demo
    #             demo_length = len(output[0])
    #             length.append(demo_length)



    # Find the total lengths now
    # whole_length = len(tactile_indices)
    # desired_len = int((duration / 120) * whole_length)

    # data = dict(
    #     observations = observations,
    #     actions = actions,
    #     images = image,
    #     length = length
    # )

    data = dict(
    #     tactile = dict(
    #         indices = tactile_indices[:desired_len],
    #         values = tactile_values
    #     ),
        allegro_joint_states = dict(
            # indices = allegro_indices[:desired_len], 
            values = allegro_joint_positions,
            torques = allegro_joint_torques
        ),
    #     allegro_tip_states = dict(
    #         indices = allegro_indices[:desired_len], 
    #         values = allegro_tip_positions
    #     ),
        allegro_actions = dict(
            # indices = allegro_action_indices[:desired_len],
            values = allegro_actions
        ),
        length = length,
        demo = demo
    #     kinova = dict( 
    #         indices = kinova_indices[:desired_len], 
    #         values = kinova_states
    #     ), 
    #     image = dict( 
    #         indices = image_indices[:desired_len]
    #     )
    )

    return data
# roots = glob.glob(f'/home/yinlongdai/Desktop/holobot_data/cube_flipping/demonstration_*')
# data = load_data(roots, demos_to_use=[1,2,3,4,5,6,11], duration=120)
# print(data['length'][3][1])

























def get_image_stats(len_image_dataset, image_loader):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum    += inputs.sum(axis = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    # pixel count
    count = len_image_dataset * 480 * 480

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

def load_dataset_image(data_path, demo_id, image_id, view_num, transform=None):
    roots = glob.glob(f'{data_path}/demonstration_*')
    roots = sorted(roots)
    image_root = roots[demo_id]
    image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(view_num, str(image_id).zfill(5)))
    img = loader(image_path)
    if not transform is None:
        img = transform(img)
        img = torch.FloatTensor(img)
    return img

# Taken from https://github.com/NYU-robot-learning/multimodal-action-anticipation/utils/__init__.py#L90
def batch_indexing(input, idx):
    """
    Given an input with shape (*batch_shape, k, *value_shape),
    and an index with shape (*batch_shape) with values in [0, k),
    index the input on the k dimension.
    Returns: (*batch_shape, *value_shape)
    """
    batch_shape = idx.shape
    dim = len(idx.shape)
    value_shape = input.shape[dim + 1 :]
    N = batch_shape.numel()
    assert input.shape[:dim] == batch_shape, "Input batch shape must match index shape"
    assert len(value_shape) > 0, "No values left after indexing"

    # flatten the batch shape
    input_flat = input.reshape(N, *input.shape[dim:])
    idx_flat = idx.reshape(N)
    result = input_flat[np.arange(N), idx_flat]
    return result.reshape(*batch_shape, *value_shape) 