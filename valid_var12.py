## to be completed later ####



import logging
import pathlib
import random
import shutil
import time
import h5py
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
import os
from data import transforms as T
from data.data_loader2 import SliceData , DataTransform
from models.models import UnetModel,DataConsistencyLayer , SensitivityModel , network,network , _NetG_lite ,SSIM, loss_prediction_module
from tqdm import tqdm
import data.transforms as T
import csv
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):
    print("data taken from:",args.data_path)
    dev_data = SliceData(
        root=str(args.data_path),
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        acceleration=args.acceleration
    )
    return dev_data

def create_data_loaders(args):
    dev_data = create_datasets(args)

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    
    return dev_loader 


def reconstruct(args,  model ,  data_loader):
    
    model.eval()

    with torch.no_grad():
        lo_list = []
        for iter, data in enumerate(tqdm(data_loader)):
            
            ksp_us,img_us,img_gt_np,_,mask,maxi,fname = data
                
            # input_kspace = input_kspace.to(args.device)
            # inp_mag = mag_us.unsqueeze(1).to(args.device)
            # tgt_mag = mag_gt.unsqueeze(1).to(args.device)
            # inp_pha = pha_us.unsqueeze(1).to(args.device)
            # tgt_pha = pha_gt.unsqueeze(1).to(args.device)
            # target = target.unsqueeze(1).to(args.device)
            ksp_us = ksp_us.to(args.device)
            # sens = sens.to(args.device)
            mask = mask.to(args.device)
            img_us = img_us.to(args.device)
            # img_gt = img_gt.to(args.device)
            img_gt_np = img_gt_np.unsqueeze(0).to(args.device).float()
            
            ###### for ZF reconstructions  ###### 
            # out_mag = img_us_np
                        
            out_mag = out_mag.cpu()
            # print("out_mag",out_mag.shape)          
            fname = (pathlib.Path(str(fname)))
            parts = list(fname.parts)
            
            f_slice = parts[-1][:-2]
            
            
            # print("parts",parts[-1][:-2])
            # 
            parts[-3] = 'Recons/varnet/'+ str(args.channel) + '_channel/acc_' + str(args.acceleration) + 'x/' + str(args.model) +'/reconstructions'
            parts.pop(-2)
            # print("parts",parts[-2])
            path=pathlib.Path(*parts) 
            
            
            path = str(path)[2:-3]
            # print("out_mag",out_mag.shape)

            
            # print("recons_path",args.recons_path)
            if not os.path.exists(args.recons_path + '/reconstructions'): 
                os.makedirs(args.recons_path + '/reconstructions')
            with h5py.File(str(path), 'w') as f:
                # print(path)
                f.create_dataset("Recons", data=out_mag.squeeze(0)*maxi.float())
                # f.create_dataset("loss_pred", data=out_lo)
                
            
            
            # lo_list.append([f_slice , out_lo])
            lo_list.append([f_slice ])
            
        # print("len",len(lo_list))
        # if not os.path.exists(str(args.recons_path) + '/loss_pred'): 
        #     os.makedirs(args.recons_path+ '/loss_pred')
        # lo_file = str(args.recons_path) + '/loss_pred/loss_pred.csv'
        
        # pred_df = pd.DataFrame(lo_list, columns=['filename', 'pred_loss'])
        # pred_df.to_csv(lo_file)
        
        
        # with open(lo_file, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(lo_list)
        print("Reconstructions saved @ :",str(path)[:-23])


def build_model(args):
    
    wacoeff = 0.1
    dccoeff = 0.1
    cascade = 5   
    sens_chans = 8
    sens_pools = 4

    model = architecture(dccoeff, wacoeff, cascade,sens_chans, sens_pools).to(args.device)


    return  model  


def load_model(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model_vs = build_model(args)
    if args.data_parallel:

        model_vs = torch.nn.DataParallel(model_vs)

    model_vs.load_state_dict(checkpoint['model_vs'])

    optimizer = build_optim(args, model_vs.parameters())

    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    return checkpoint,  model_vs ,   optimizer


def main(args):
    
    dev_loader  = create_data_loaders(args)
    # model_vs, model_lo = load_model(args.model_path) 
    model_vs,model_sens = load_model(args.model_path)
    # reconstruct(args,  model_vs ,model_lo,  dev_loader)
    reconstruct(args,  model_vs , model_sens ,  dev_loader)



def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--model-path', type=str, help='Path where model is present')
    parser.add_argument('--data-path', type=pathlib.Path,help='Path to the dataset')
    parser.add_argument('--recons-path', type=str, help='Path where reconstructions are to be saved')
    parser.add_argument('--acceleration', type=int,help='Ratio of k-space columns to be sampled. 5x or 10x masks provided')
    parser.add_argument('--channel', type=int,help='whether 12 channels data or 32 channels data')
    parser.add_argument('--dropout', type=float,help='% of dropping of nodes')
    parser.add_argument('--model', type=str, default='model',help='Name of architecture')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
  
    main(args)



