import glob
import numpy as np
import os
import torch
import torchvision.transforms as T 

from torch.utils import data
from torchvision.datasets.folder import default_loader as loader 

# from tactile_learning.tactile_data import TactileImage
from utils.data import load_data
# from tactile_learning.utils import crop_transform, VISION_IMAGE_MEANS, VISION_IMAGE_STDS
from utils.constant import VISION_IMAGE_MEANS, VISION_IMAGE_STDS
from utils.augmentation import crop_transform
from datasets.data_sampler import data_sampler

# class TactileVisionActionDataset(data.Dataset):
class VisionActionDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        # tactile_information_type,
        # tactile_img_size,
        vision_view_num
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        # self.roots = glob.glob(f'{data_path}/demos/hammer-v2') 
        self.roots = sorted(self.roots)

        self.data = load_data(self.roots, demos_to_use=[])
        # assert tactile_information_type in ['stacked', 'whole_hand', 'single_sensor'], 'tactile_information_type can either be "stacked", "whole_hand" or "single_sensor"'
        # self.tactile_information_type = tactile_information_type
        self.vision_view_num = vision_view_num

        self.vision_transform = T.Compose([
            T.Resize((480,640)), #size may change
            T.Lambda(self._crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

        # Set the indices for one sensor
        # if tactile_information_type == 'single_sensor':
        #     self._preprocess_tactile_indices()
    
        # self.tactile_img = TactileImage(
        #     tactile_image_size = tactile_img_size,
        #     shuffle_type = None
        # )

    def _crop_transform(self, image):
         return crop_transform(image, self.vision_view_num)

    # def _preprocess_tactile_indices(self):
    #     self.tactile_mapper = np.zeros(len(self.data['tactile']['indices'])*15).astype(int)
    #     for data_id in range(len(self.data['tactile']['indices'])):
    #         for sensor_id in range(15):
    #             self.tactile_mapper[data_id*15+sensor_id] = data_id # Assign each finger to an index basically

    # def _get_sensor_id(self, index):
    #     return index % 15

    def __len__(self):
        # if self.tactile_information_type == 'single_sensor':
        #     return len(self.tactile_mapper)
        # else: 
        #     return len(self.data['tactile']['indices'])

        ##Since we are not indexing for now, we the length is 
        ##the number of images dumped
            #check the number of images dumped in root 
        length = 0 
        for i in range(len(self.data['length'])):
            length += self.data['length'][i][1]
        return length
        
    # def _get_proper_tactile_value(self, index):
    #     if self.tactile_information_type == 'single_sensor':
    #         data_id = self.tactile_mapper[index]
    #         demo_id, tactile_id = self.data['tactile']['indices'][data_id]
    #         sensor_id = self._get_sensor_id(index)
    #         tactile_value = self.data['tactile']['values'][demo_id][tactile_id][sensor_id]
            
    #         return tactile_value
        
    #     else:
    #         demo_id, tactile_id = self.data['tactile']['indices'][index]
    #         tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
            
    #         return tactile_values

    def _get_image(self, index):
        #This is what the length is for

        # demo_id, image_id = self.data['image']['indices'][index]
        # image_root = self.roots[demo_id]
        # image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.vision_view_num, str(image_id).zfill(5)))
        ##Check which demo the index belongs to
        # print("-------------------------------------------------------- demos: {}".format(self.data['length']))
        # print("--------------------------------------------------previous index{}".format(index))
        for i in range(len(self.data['length'])):
            if index < self.data['length'][i][1]:
                break
            else:
                index -= self.data['length'][i][1]
        # print("-------------------------------------------------current index{}".format(index))
        ##Get demo_id
        demo_id = self.data['length'][i][0]
        ## Get the image from the root
        image_path = self.roots[demo_id] + '/cam_{}_rgb_images/frame_{}.png'.format(self.vision_view_num, str(index).zfill(5))
        # img = self.data['images'][index]
        img = self.vision_transform(loader(image_path))
        # img = self.vision_transform(img)
        return torch.FloatTensor(img)

    # def _get_tactile_image(self, tactile_values):
    #     return self.tactile_img.get(
    #         type = self.tactile_information_type,
    #         tactile_values = tactile_values
    #     )

    # Gets the kinova states and the commanded joint states for allegro
    # def _get_action(self, index):
    #     demo_id, allegro_action_id = self.data['allegro_actions']['indices'][index]
    #     allegro_action = self.data['allegro_actions']['values'][demo_id][allegro_action_id]

    #     _, kinova_id = self.data['kinova']['indices'][index]
    #     kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

    #     total_action = np.concatenate([allegro_action, kinova_action], axis=-1)
    #     return torch.FloatTensor(total_action) # These values are already quite small so we'll not normalize them

    # def _get_action (self, index):
    #     action = self.data['actions'][index]
    #     return torch.FloatTensor(action) # do we have to normalize this?

    def __getitem__(self, index):
        # tactile_value = self._get_proper_tactile_value(index)
        # tactile_image = self._get_tactile_image(tactile_value)

        vision_image = self._get_image(index)

        # action = self._get_action(index)
        
        # return vision_image, action
        return vision_image
    




class TemporalVisionJointDiffDataset(data.Dataset): # Class to train an encoder that holds the tempoeral information
    def __init__(
        self,
        data_path,
        vision_view_num,
        vision_img_size,
        frame_diff # Number of frame differences
    ):

        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.view_num = vision_view_num
        self.img_size = vision_img_size
        self.frame_diff = frame_diff

        self.vision_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(self._crop_transform),
            T.Resize(vision_img_size),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

        self.sampled_data = data_sampler(data_path, vision_view_num)
        
    def _crop_transform(self, image):
        return crop_transform(image, camera_view=self.view_num, image_size=self.img_size)

    # def __len__(self):
    #     return len(self.data['image']['indices']) - self.frame_diff
    def __len__(self):
        return len(self.sampled_data) - self.frame_diff 
    
    # def _get_joint_state(self, index, kinova_index=None):
    #     demo_id, allegro_id = self.data['allegro_joint_states']['indices'][index]
    #     allegro_action = self.data['allegro_joint_states']['values'][demo_id][allegro_id]
    #     _, kinova_id = self.data['kinova']['indices'][index]
    #     kinova_action = self.data['kinova']['values'][demo_id][kinova_id]

    #     total_state = np.concatenate([allegro_action, kinova_action], axis=-1)
    #     return total_state

    def _get_joint_state(self, index, kinova_index=None):
        demo_id, allegro_id = self.sampled_data[index][0], self.sampled_data[index][2]
        allegro_action = self.data['allegro_joint_states']['values'][demo_id][allegro_id]

        total_state = np.array([allegro_action])
        return total_state
    
    
    # Gets the kinova states and the commanded joint states for allegro
    def _get_joint_diff(self, index):
        curr_joint_state = self._get_joint_state(index)
        closest_joint_state = self._find_the_closest_last_frame(index, data_type='kinova')
        next_joint_state = self._get_joint_state(closest_joint_state)

        joint_state_diff = next_joint_state - curr_joint_state
        return torch.FloatTensor(joint_state_diff)

    # def _get_image(self, index):
    #     demo_id, image_id = self.data['image']['indices'][index]
    #     image_root = self.roots[demo_id]
    #     image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.view_num, str(image_id).zfill(5)))
    #     img = self.vision_transform(loader(image_path))
    #     return torch.FloatTensor(img) 
    def _get_image(self, index):
        demo_id, image_id = self.sampled_data[index][0], self.sampled_data[index][1]
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.view_num, str(image_id).zfill(5)))
        img = self.vision_transform(loader(image_path))
        return torch.FloatTensor(img) 
    
    # def _find_the_closest_last_frame(self, index, data_type):
    #     old_demo_id, _ = self.data[data_type]['indices'][index]
    #     for i in range(index+1, index+self.frame_diff, 1):
    #         demo_id, _ = self.data[data_type]['indices'][i]
    #         if demo_id != old_demo_id:
    #             return i-1
         
    #     return index + self.frame_diff
    def _find_the_closest_last_frame(self,index,data_type):
        old_demo_id = self.sampled_data[index][0]
        ##check if we have jumped to another demo
        for i in range(index+1, index+self.frame_diff, 1):
            demo_id = self.sampled_data[i][0]
            if demo_id != old_demo_id:
                return i-1
            
        return index + self.frame_diff


    def __getitem__(self, index):
        curr_image = self._get_image(index)
        closest_img_id = self._find_the_closest_last_frame(index, data_type = 'image')
        next_image = self._get_image(closest_img_id)

        joint_diff = self._get_joint_diff(index)

        return curr_image, next_image, joint_diff
# if __name__ == '__main__':
#     dataset = 