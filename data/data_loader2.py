import torch
import pathlib
import random
import data.transforms as T
import h5py
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from bart import bart
import math
import random

   
class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform,no_of_vol,acceleration):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace',
                filename',  'sensitivity maps', and 'acclearation' as inputs. 

            sample_rate : A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
            acceleration: Whether to use 5x US data or 10x US data
        """
        self.acceleration = acceleration

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())    # total number of all files ie 7332

        
        # if sample_rate < 1:
        #     random.shuffle(files)
        #     num_files = round(len(files) * sample_rate)
        #     files = files[:num_files]


        # if sample_rate < 1:
        # random.shuffle(files)
        num_files = len(files) # * sample_rate)
        files = files[:156*no_of_vol]     # here sample_rate is the number of volumes to be
            
        for fname in sorted(files):

                self.examples.append(str(fname)) 

            
    def __len__(self):

        return len(self.examples)
    

    def __getitem__(self, i):

        fname = self.examples[i]
        
        with h5py.File(fname, 'r') as data:

            ksp = data['kspace'][()]
            sens = data['sensitivity'][()]
        

        return self.transform (ksp, fname, sens, self.acceleration)
    
    
    
    
class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):   
        """
        Not required !
        """
    
    def __call__(self,ksp_cmplx,fname,sensitivity,acceleration):
        """
        Args:
            ksp_cmplx (numpy.array): Input k-space of the multi-coil data
            fname (str): File name
            sensitivity(numpy.array): ENLIVE sensitivity maps
            acceleartion: whether to train for 5x US ksp or 10x US kspace

        """

        sens_t = T.to_tensor(sensitivity)
        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_t = ksp_t.permute(2,0,1,3)

        img_gt_np = T.root_sum_of_squares(T.complex_abs(T.ifft2_np(ksp_t)))
        
        
        if acceleration == 5:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x180.npy")
        
        elif acceleration == 10:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x180.npy")
        
        

        randint = random.randint(0,99)                   #to get a random mask everytime ! 
        mask = sp_r5[randint]
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()
        
        ksp_us = torch.where(mask == 0, torch.Tensor([0]), ksp_t)

        img_us_np = T.root_sum_of_squares(T.complex_abs(T.ifft2_np(ksp_us)))
        img_us = T.ifft2(ksp_us)
        img_us_rss = T.root_sum_of_squares(T.complex_abs(T.ifft2(ksp_us)))


        maxi = img_us_rss.max().float()

        return   ksp_us/maxi ,  img_us/maxi  , img_us_rss/maxi , 100.0*img_us_np/maxi , 100.0*img_gt_np/maxi ,sens_t, mask ,maxi,fname




class DataTransform_rotnet:
    """
    Data Transformer for the pretext task of RotNet.
    """

    def __init__(self):   
        """
        Not required !
        """
    
    def __call__(self,ksp_cmplx,fname,sensitivity,acceleration):
        """
        Args:
            kspace (numpy.array): Input k-space of the multi-coil data
            fname (str): File name
            sensitivity maps (numpy.array): ENLIVE sensitivity maps
            acceleartion: whether to train for 5x US ksp or 10x US kspace

        """
    ## comment lines 89,90,91,92,93 
        # fname = '/media/student1/RemovableVolume/calgary/12_channels_218_180/Train/e16971s3_P23040.7.100.h5' 
        # with h5py.File(fname, 'r') as data:

        #     ksp_cmplx = data['kspace'][()]
        #     sensitivity = data['sensitivity'][()]
    
    
    ## start from here    
        # sens_t = T.to_tensor(sensitivity)
       
        # mask_sampling = ~( np.abs(ksp_cmplx).sum( axis = (-1) ) == 0)
        # mask = 1.0*mask_sampling
        # mask = torch.from_numpy(mask)
        
        # mask = (torch.stack((mask,mask),dim=-1)).float()
        
        sens_t = T.to_tensor(sensitivity)
        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_t = ksp_t.permute(2,0,1,3)

        img_gt_rss = T.root_sum_of_squares(T.complex_abs(T.ifft2(ksp_t)))
        
        # maxi = T.zero_filled_reconstruction(ksp_cmplx_rot).max()*100.0
        if acceleration == 5:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x180.npy")
        
        elif acceleration == 10:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x180.npy")
        
        

        randint = random.randint(0,99)                   #to get a random mask everytime ! 
        mask = sp_r5[randint]
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()

        ksp_us = torch.where(mask == 0, torch.Tensor([0]), ksp_t)

        
        degrees = ('0' , '90' , '180' , '270')
        randint = random.randint(0,3)                   #to get a random rotation [0,90,180,270] everytime ! 
        deg = degrees[randint]

        ksp_us = T.rotation(ksp_us,deg) #.transpose(1,2,0)

        mask = T.rotation(mask.unsqueeze(0),deg).squeeze(0)

        img_us = T.ifft2(ksp_us)

        img_us_rss = T.root_sum_of_squares(T.complex_abs(T.ifft2(ksp_us)))
        maxi = img_us_rss.max().float()


        return   ksp_us/maxi , img_us/maxi,  img_us_rss/maxi , img_gt_rss/maxi ,  randint ,   mask , sens_t ,  maxi , fname
        # return   ksp_us, ksp_t ,  img_us, img_gt , img_us_np , img_gt_np , sens_t , mask ,img_us.max(),img_us_np.max(),fname



class DataTransform_slicemiss:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):   
        """
        Not required !
        """
    
    def __call__(self,ksp_cmplx,fname,sensitivity,acceleration):
        """
        Args:
            ksp_cmplx (numpy.array): Input k-space of the multi-coil data
            fname (str): File name
            sensitivity(numpy.array): ENLIVE sensitivity maps
            acceleartion: whether to train for 5x US ksp or 10x US kspace

        """

        sens_t = T.to_tensor(sensitivity)
        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_t = ksp_t.permute(2,0,1,3)

        img_gt = T.ifft2(ksp_t)
        img_gt_np = T.root_sum_of_squares(T.complex_abs(T.ifft2_np(ksp_t)))
        
        
        if acceleration == 5:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x180.npy")
        
        elif acceleration == 10:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x180.npy")
        
        

        randint = random.randint(0,99)                   #to get a random mask everytime ! 
        mask = sp_r5[randint]
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()
        
        ksp_us = torch.where(mask == 0, torch.Tensor([0]), ksp_t)
        
        r_int = randint = random.randint(0,11) 
        ksp_us0 = ksp_us.clone()
        ksp_us0[r_int,:,:,:] = 1e-8


        # degrees = ('0' , '90' , '180' , '270')
        # randint = random.randint(0,3)                   #to get a random rotation [0,90,180,270] everytime ! 
        # deg = degrees[randint]

        # ksp_us0 = T.rotation(ksp_us0,deg) #.transpose(1,2,0)

        # mask = T.rotation(mask.unsqueeze(0),deg).squeeze(0)

        img_us_np = T.root_sum_of_squares(T.complex_abs(T.ifft2_np(ksp_us0)))
        img_us = T.ifft2(ksp_us0)
        img_us_rss = T.root_sum_of_squares(T.complex_abs(T.ifft2(ksp_us0)))


        maxi = img_us_rss.max().float()

        return   ksp_us0/maxi ,  img_us/maxi  , img_us_rss/maxi , 100.0*img_us_np/maxi , 100.0*img_gt_np/maxi ,sens_t, mask ,maxi,fname,r_int,img_gt/maxi



class DataTransform_multitask:
    """
    Data Transformer for pre-training the models slicemiss+ rotnet
    """

    def __init__(self):   
        """
        Not required !
        """
    
    def __call__(self,ksp_cmplx,fname,sensitivity,acceleration):
        """
        Args:
            ksp_cmplx (numpy.array): Input k-space of the multi-coil data
            fname (str): File name
            sensitivity(numpy.array): ENLIVE sensitivity maps
            acceleartion: whether to train for 5x US ksp or 10x US kspace

        """

        sens_t = T.to_tensor(sensitivity)
        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_t = ksp_t.permute(2,0,1,3)

        img_gt = T.ifft2(ksp_t)
        img_gt_np = T.root_sum_of_squares(T.complex_abs(T.ifft2_np(ksp_t)))
        
        
        if acceleration == 5:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x180.npy")
        
        elif acceleration == 10:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x180.npy")
        
        

        randint = random.randint(0,99)                   #to get a random mask everytime ! 
        mask = sp_r5[randint]
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()
        
        ksp_us = torch.where(mask == 0, torch.Tensor([0]), ksp_t)
        
        r_int = randint = random.randint(0,11) 
        ksp_us0 = ksp_us.clone()
        ksp_us0[r_int,:,:,:]=0 + 1e-8

        img_us_np = T.root_sum_of_squares(T.complex_abs(T.ifft2_np(ksp_us0)))
        img_us = T.ifft2(ksp_us0)
        img_us_rss = T.root_sum_of_squares(T.complex_abs(T.ifft2(ksp_us0)))


        maxi = img_us_rss.max().float()

        return   ksp_us0/maxi ,  img_us/maxi  , img_us_rss/maxi , 100.0*img_us_np/maxi , 100.0*img_gt_np/maxi , randint, sens_t, mask ,maxi,fname,r_int,img_gt/maxi
        

