import h5py
import numpy as np
import data.transforms as T
from matplotlib import pyplot as plt
import torch
import math
from torch.nn import functional as F
from collections import namedtuple
from pathlib import Path
from torch import nn
import cv2 as  cv
import torch.nn as nn
import pathlib
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
from models.models import  architecture_unet
import random
import os
import numpy as np

class SliceData(Dataset):
    
    def __init__(self,transform,root):

        self.transform = transform
        # self.root = root
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        
        sample_rate = 1.0    ## no need of sampling, remove it later !!
        if sample_rate < 1:
            random.shuffle(files)
        num_files = round(len(files) * sample_rate)
        files = files[:num_files]
        
        
        
        for fname in sorted(files):

                num_slices = 256
                self.examples += [(fname, slice) for slice in range(50,num_slices-50)]

            
    def __len__(self):

        return len(self.examples)
    

    def __getitem__(self, i):
        data_path, slice = self.examples[i]

        # print("data_path",data_path)

        # print("slice",slice)
        with h5py.File(data_path, 'r') as data:

            zf_kspace = data['kspace'][()]
            mask_sampling = ~( np.abs(zf_kspace).sum( axis = (0, -1) ) == 0)
            mask_np = 1.0*mask_sampling
            ksp = zf_kspace[slice]

        
        return self.transform (ksp, mask_np,data_path,slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):
        """
        We will, we will rock you !!
        
        """    
    def __call__(self,ksp,mask,fname,slice):
        
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()
        
        ksp_cmplx = ksp[:,:,::2] + 1j*ksp[:,:,1::2]
       

        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_us= ksp_t.permute(2,0,1,3)

        img_us = T.ifft2(ksp_us)

        img_us_rss = T.root_sum_of_squares(T.complex_abs(T.ifft2(ksp_us)))
        maxi = img_us_rss.max().float()


        return ksp_us/maxi ,  img_us/maxi , img_us_rss/maxi  , mask, fname.name, slice,maxi 


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)    

def create_data_loaders(root):
    
    dev_data = SliceData(transform=DataTransform(),root = root)
    
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    
    return dev_loader


def run_submission(model, data_loader):
    

    model.eval()
    
    reconstructions = defaultdict(list)
    for iter, data in enumerate(tqdm(data_loader)):
        
        ksp_us,img_us,img_us_np,mask,fname, slice, maxi = data

        ksp_us = ksp_us.cuda()
        img_us = img_us.cuda()
 
        mask = mask.cuda()
        maxi = maxi.float().cuda()

        out,_,_ = model(img_us,ksp_us,mask)
        # print("out_max=",out.max())
        out = out.float()*maxi/100.0


        for i in range(1):
            reconstructions[fname[i]].append((slice[i].detach().cpu().numpy(), out[i].detach().cpu().numpy()))
  
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
        }
    return reconstructions


def build_model(args):
    
    wacoeff = 0.1
    dccoeff = 0.1
    cascade = 12 #args.cascade   
    sens_chans = 8
    sens_pools = 4

    model = architecture_unet(dccoeff, wacoeff, cascade,sens_chans, sens_pools).cuda()


    return  model  


def load_model(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    print("check_point file = ",checkpoint_file)
    model = build_model(args)

    model.load_state_dict(checkpoint['model'])

    
    print("trained model loaded.....")
    
    return model




def main(test_data_path,model_path,out_dir):

    print("data taken from = ",test_data_path)
    data_loader = create_data_loaders(test_data_path)
    print("dataloaders readdy.....")
    model = load_model(model_path)
    reconstructions = run_submission(model, data_loader)
    save_reconstructions(reconstructions, Path(out_dir))
    print()
    print("Reconstructions saved @ :",out_dir)



if __name__ == '__main__':
    


                                                #########  12-channel   #######

    # VOLUME = '5'
    # ACTION = 'scratch'

    # test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_12_channel/Test-R=5/"
    # model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/" + ACTION +"/acc_5x/" + VOLUME + "_volume/12_cascade/0.001_lr/best_model.pt"
    # out_dir = "/media/student1/RemovableVolume/calgary_new/" + VOLUME + ACTION +"_volume/Track01/12-channel-R=5"

    
    VOLUME = '5'
    ACTION = 'finetune'
    PRETEXT = 'slicemiss'
    LAYER='0'

    test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_12_channel/Test-R=5/"
    model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/slicemiss/finetune/acc_5x/0_layer/0.001_lr/5_volume/best_model.pt"
    out_dir = "/media/student1/RemovableVolume/calgary_new/" + PRETEXT + "/" + VOLUME +"_volume/Track01/12-channel-R=5"



                                                #########  32-channel  #######
    # VOLUME = '5'
    # ACTION = 'scratch'
    
    # test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_32_channel/Test-R=5/"
    # model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/" + ACTION +"/acc_5x/" + VOLUME + "_volume/12_cascade/0.001_lr/best_model.pt"
    # out_dir = "/media/student1/RemovableVolume/calgary_new/" + VOLUME + ACTION +"_volume/Track02/32-channel-R=5"

    # VOLUME = '5'
    # ACTION = 'finetune'
    # PRETEXT = 'rotnet'

    # test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_32_channel/Test-R=5/"
    # model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/" + PRETEXT + "/" + ACTION +"/acc_5x/0_layer/0.001_lr/" + VOLUME + "_volume/best_model.pt"
    # out_dir = "/media/student1/RemovableVolume/calgary_new/" + PRETEXT + "/" + VOLUME + "_volume/Track02/32-channel-R=5"
        
    
    main(test_data_path,model_path,out_dir)    
    
