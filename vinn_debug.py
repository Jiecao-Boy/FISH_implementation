import glob
import h5py
import hydra
import mmap
import numpy as np
import os
import pickle
import torch 
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.utils.data as data 
import random

from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from utils.constant import *
from model.utils import *
# from model.custom import *
# from datasets.tactile_vision import *
# from tactile_learning.deployment.load_models import * 
from deployer.nnbuffer import NearestNeighborBuffer
from model.knneighbor import ScaledKNearestNeighbors
from utils.visualization import *
from utils.tactile_image import *
from utils.data import load_data, load_dataset_image
from torchvision.transforms.functional import crop
from holobot.samplers.allegro import AllegroSampler

TEST_DIR = '/scratch/yd2032/Desktop/holobot_data/cube_flipping'
TEST_ROOTS = sorted(glob.glob(f'{TEST_DIR}/demonstration_*'))
REPR_DIR = '/scratch/yd2032/Desktop/holobot_data/cube_flipping'
REPR_ROOTS = sorted(glob.glob(f'{REPR_DIR}/demonstration_*'))

#ALEXMNET_TACK_OUT_DIR_NOT_PRETRAINED ??
class TINNStarter: 
    def __init__(
        self,
        image_encoder_out_dir,
        # image_nontrained=False,
        view_num = 0,
        test_demos = [],
        repr_demos =[]
    ):
        # os.enciron["MASTER_ADDR"] = "172.24.71.211"
        # os.environ["MASTER_PORT"] = "29505"

        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')

        self.view_num = view_num
        self.image_cfg, self.image_encoder, self.image_transform = self._init_encoder_info(self.device, image_encoder_out_dir) 
        self.inv_image_transform = self._get_inverse_image_norm()

        self.test_data = load_data(TEST_ROOTS, demos_to_use=test_demos)
        # print('TEST_DATA LEN: {}'.format(sum(self.test_data['length'])))
        self.repr_data = load_data(REPR_ROOTS, demos_to_use=repr_demos)

        IMAGE_RPER_SIZE = 512

        #get test representations
        self.test_repr =   all_representations = np.zeros((
                          0, 528
                          ))
        for demo_id, root in enumerate (TEST_ROOTS):
            if demo_id == 4:
                print("demo id:{}".format(demo_id))
                self.test_sampler = AllegroSampler(root, [0], 'rgb', 0.01)
                self.test_sampler.sample_data()
                ##store the image and state information: 
                for index in range(len(self.test_sampler.sampled_robot_states)):
                     ## states
                    representation = self.test_sampler.sampled_robot_states[index]
                    ## image and preproccessing
                    dset_img = load_dataset_image(TEST_DIR, demo_id, self.test_sampler.sampled_rgb_frame_idxs[0][index], 0)
                    img = torch.FloatTensor(self.image_transform(dset_img)).to(self.device)
                    image = self.image_encoder(img.unsqueeze(dim=0)) # Add a dimension to the first axis so that it could be considered as a batch
                    image = image.detach().cpu().numpy().squeeze()
                    representation = np.concatenate([image, representation], axis=0)
                    self.test_repr=np.vstack((self.test_repr, representation))
        print("---------------------------------------------------------num of frames sampled {}, demo: {}".format(len(self.test_sampler.sampled_rgb_frame_idxs[0]),demo_id))


        #get all representations
        self.all_repr =   all_representations = np.zeros((
                          0, 528
                          ))
        self.demo_img = []
        for demo_id, root in enumerate (TEST_ROOTS):
            if demo_id in [0,1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]:
                continue
            print("demo id:{}".format(demo_id))
            self.sampler = AllegroSampler(root, [0], 'rgb', 0.01)
            self.sampler.sample_data()
            ##store the image and state information:  
            for index in range(len(self.sampler.sampled_robot_states)):
                 ## states
                representation = self.sampler.sampled_robot_states[index]
                ## image and preproccessing
                dset_img = load_dataset_image(TEST_DIR, demo_id, self.sampler.sampled_rgb_frame_idxs[0][index], 0)
                img = torch.FloatTensor(self.image_transform(dset_img)).to(self.device)
                image = self.image_encoder(img.unsqueeze(dim=0)) # Add a dimension to the first axis so that it could be considered as a batch
                image = image.detach().cpu().numpy().squeeze()
                representation = np.concatenate([image, representation], axis=0)
                self.all_repr=np.vstack((self.all_repr, representation))
                #save the demo_id and image_id as well
                self.demo_img.append((demo_id, self.sampler.sampled_rgb_frame_idxs[0][index]))

        print("---------------------------------------------------------num of frames sampled {}, demo: {}".format(len(self.sampler.sampled_rgb_frame_idxs[0]),demo_id))
        self.image_knn = ScaledKNearestNeighbors(
            self.all_repr[:,:512],
            self.all_repr[:,:512],
            ['image'],
            [1],
        )

        self.K = 5
        self.TEST_NUM = len(self.test_sampler.sampled_rgb_frame_idxs[0])
    




    def _init_encoder_info(self, device, out_dir): # encoder_type: either image or tactile
        cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        model_path = os.path.join(out_dir, 'saved_models/byol_encoder_best.pt')
        encoder = load_model(cfg, device, model_path)

        transform = T.Compose([
                T.Resize((480,640)),
                T.Lambda(self._crop_transform),
                T.ToTensor(),
                T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
            ])
        
        return cfg, encoder, transform
    
    def _get_inverse_image_norm(self):
        np_means = np.array(VISION_IMAGE_MEANS)
        np_stds =np.array(VISION_IMAGE_STDS)

        inv_normalization_transform = T.Compose([
            T.Normalize(mean= [0,0,0],std = 1/np_stds),
            T.Normalize(mean = -np_means, std = [1,1,1])
        ])
        return inv_normalization_transform

    
    def _crop_transform(self, image):
        if self.view_num == 0:
            return crop(image, 0,0,480,480)
        elif self.view_num == 1:
            return crop(image, 0,120,480,640)
        







    def get_all_neighbors(self, k):
        self.K = k
        repr_size = 512
        self.all_neighbors = np.zeros((
            len(self.test_repr), self.K
        )).astype(int)
        self.neighbor_data = []
        for repr_id, test_repr in enumerate(self.test_repr):
            curr_neighbor_data = self._get_one_neighbor_for_all_types(test_repr)
            self.all_neighbors[repr_id,:] = curr_neighbor_data['image']['ids'][:]
            self.neighbor_data.append(curr_neighbor_data)
            
    def _get_one_neighbor_for_all_types(self, test_repr):
        _, image_neighbor_ids, image_neighbor_dists = self.image_knn.get_k_nearest_neighbors(test_repr[:512], k=self.K)

        neighbors = dict(
            image = dict(ids = image_neighbor_ids, dists = image_neighbor_dists)
        )
        return neighbors



        
    def plot_all_neighbors(self, repr_types):
        if isinstance(repr_types, list):
            all_nn_idxs = self.all_neighbors[:,:,2]
            repr_type_str = 'all'
        elif repr_types == 'image':
            all_nn_idxs = self.all_neighbors[:,:]
            repr_type_str = 'image'
        elif repr_types == 'tactile':
            all_nn_idxs = self.all_neighbors[:,:,1]
            repr_type_str = 'tactile'

        repr_indices = random.choices(range(self.TEST_NUM), k=min(30,self.TEST_NUM))
        TEST_NUM = len(range(0,len(repr_indices)))

        figsize=((self.K+1)*4, TEST_NUM*8)
        fig, axs = plt.subplots(figsize=figsize, nrows=TEST_NUM*2, ncols=self.K+1)
        axs[0][0].set_title("Actual")


        
        for i in range(self.K):
            axs[0][i+1].set_title(f"{i+1}th Neighbor")
        
        ## plot the representation it self
        for axs_id, test_id in enumerate(repr_indices):
            test_image_id = self.test_sampler.sampled_rgb_frame_idxs[0][test_id]
            image = load_dataset_image(TEST_DIR, demo_id=4, image_id = test_image_id,view_num = 0)
            img = torch.FloatTensor(self.image_transform(image)).to(self.device)
            test_img = self.inv_image_transform(img).cpu().numpy().transpose(1,2,0)
            test_img_cv2 = test_img*255

            self.plot_state(
                axs[2*axs_id][0], test_img_cv2
            )
            ## plot the corresponding neighbors
            print(len(self.sampler.sampled_rgb_frame_idxs[0]))
            for k in range(self.K):
                nn_id = all_nn_idxs[test_id, k]
                print("-----------------------------------------------------------{}".format(nn_id))
                # nn_image_id = self.sampler.sampled_rgb_frame_idxs[0][nn_id]
                # image = load_dataset_image(REPR_DIR,  demo_id=4, image_id = nn_image_id,view_num = 0)
                nn_image_demo = self.demo_img[nn_id]
                image = load_dataset_image(REPR_DIR, nn_image_demo[0], nn_image_demo[1], view_num = 0)
                img = torch.FloatTensor(self.image_transform(image)).to(self.device)
                nn_img = self.inv_image_transform(img).cpu().numpy().transpose(1,2,0)
                nn_img_cv2 = nn_img*255
                self.plot_state(
                    axs[2*axs_id][k+1], nn_img_cv2
                )
                ##Also plot the next action
                nn_image_demo_next = self.demo_img[nn_id+1]
                image = load_dataset_image(REPR_DIR, nn_image_demo_next[0], nn_image_demo_next[1], view_num = 0)
                img = torch.FloatTensor(self.image_transform(image)).to(self.device)
                nn_img = self.inv_image_transform(img).cpu().numpy().transpose(1,2,0)
                nn_img_cv2 = nn_img*255
                self.plot_state(
                    axs[2*axs_id+1][k+1], nn_img_cv2
                )
                axs[2*axs_id][k+1].set_xlabel('Repr_Types: {} - Dists: {}'.format(repr_types, self.neighbor_data[test_id][repr_type_str]['dists'][k]))
                axs[2*axs_id+1][k+1].set_xlabel('Repr_Types: {} - Dists: {}'.format(repr_types, self.neighbor_data[test_id][repr_type_str]['dists'][k]))
        os.makedirs("/scratch/yd2032/Desktop/byol_implementation/test_figure", exist_ok=True)
        plt.savefig(f'/scratch/yd2032/Desktop/byol_implementation/test_figure/output-1.jpg')
        
    def plot_state(self, ax, image):
        title = 'vinn_debug_dumped'
        self._dump_vision_state(None, None, title = title, vision_state = image)
        curr_state = cv2.imread(f'{title}.png')
        ax.imshow(curr_state)

    
    def _dump_vision_state(self, allegro_tip_pos, kinova_cart_pos, title='curr_state', vision_state=None):
        cv2.imwrite(f'{title}_vision.png', vision_state)
        vision_img = cv2.imread(f'{title}_vision.png')
        state_img = vision_img
        cv2.imwrite(f'{title}.png', state_img)


tinn = TINNStarter (
    image_encoder_out_dir = f'/scratch/yd2032/Desktop/byol_implementation/2023.06.01/18-47_image_byol_bs_32_epochs_500_lr_1e-05_cube_flipping',
    view_num = 0,
    test_demos = [4],
    repr_demos = [4,8,10,15,16,20]
)
print(tinn.image_encoder)
tinn.get_all_neighbors(k=5)
tinn.plot_all_neighbors(repr_types='image')
