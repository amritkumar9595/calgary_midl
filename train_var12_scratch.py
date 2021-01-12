"""
Train VarNet from scratch with 12 channels data

"""

import logging
import pathlib
import random
import shutil
import time
import wandb
import os 
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
from data import transforms as T
from data.data_loader2 import SliceData , DataTransform
from models.models import UnetModel,DataConsistencyLayer , _NetG_lite , network, SSIM ,SensitivityModel, architecture
from tqdm import tqdm



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # wandb login
# import wandb
# wandb.init()


def create_datasets(args,data_path):

    train_data = SliceData(
        root=str(data_path) + '/Train',
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        acceleration=args.acceleration
    )

    dev_data = SliceData(
        root=str(data_path) + '/Val',
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        acceleration=args.acceleration
    )
    return train_data, dev_data


def create_data_loaders(args,data_path):
    train_data, dev_data = create_datasets(args,data_path)
    display_data = [dev_data[i] for i in range(0, len(dev_data) // 16)]
    

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )


    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        shuffle = True,
        num_workers=0,
        pin_memory=True,
    )

    
    return train_loader, dev_loader , display_loader


ssim_loss = SSIM().cuda()
def train_epoch(args, epoch,model, data_loader,optimizer, writer):

    # print("entering training....")

    model.train()


    avg_loss_cmplx = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):
       
        ksp_us,img_us,img_us_np,img_gt_np,_,mask,maxi,fname = data

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

        
        # sens = model_sens(ksp_us, mask)
        # img_us =  T.combine_all_coils(img_us.squeeze(0) , sens.squeeze(0)).unsqueeze(0)
        # print("img_us_train ",img_us.max())
        # print("fname",fname)
        # print("img_us,ksp_us,sens,img_gt_np",img_us.max(),ksp_us.max(),sens.max(),img_gt_np.max())
        # print("mask in train_var12=",mask.shape)
        out,_,_ = model(img_us,ksp_us,mask)
        
        # print("out",out.shape)
        # loss_cmplx = F.mse_loss(out,img_gt_np.cuda())
        
        if(args.loss == 'SSIM'):
            loss_cmplx_mse = F.mse_loss(out,img_gt_np.cuda())
            loss_cmplx = loss_cmplx_ssim =  ssim_loss(out, img_gt_np,torch.tensor(img_gt_np.max().item()).unsqueeze(0).cuda())
        else:
            loss_cmplx =  loss_cmplx_mse = F.mse_loss(out,img_gt_np.cuda())
            loss_cmplx_ssim =  ssim_loss(out, img_gt_np,torch.tensor(img_gt_np.max().item()).unsqueeze(0).cuda())

        
        
          
        optimizer.zero_grad()

        loss_cmplx.backward()

        optimizer.step()

        avg_loss_cmplx = 0.99 * avg_loss_cmplx + 0.01 * loss_cmplx.item() if iter > 0 else loss_cmplx.item()

        writer.add_scalar('TrainLoss_cmplx_ssim', loss_cmplx.item(), global_step + iter)
        writer.add_scalar('TrainLoss_cmplx_mse', loss_cmplx_mse.item(), global_step + iter)
        # wandb.log({"Train_Loss_cmplx_ssim": loss_cmplx_ssim.item(), "Train_Loss_cmplx_mse": loss_cmplx_mse.item()})

        if iter % args.report_interval == 0:
            logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            f'Loss_cmplx = {loss_cmplx.item():.4g} Avg Loss cmplx = {avg_loss_cmplx:.4g} '
            f'Time = {time.perf_counter() - start_iter:.4f}s',
        )
        start_iter = time.perf_counter()
            
            
            
    return avg_loss_cmplx,  time.perf_counter() - start_epoch


def evaluate(args, epoch,  model ,   data_loader, writer):

    model.eval()


    losses_cmplx = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):
            
            ksp_us,img_us,img_us_np,img_gt_np,_,mask,maxi,fname = data
            
            # inp_mag = mag_us.unsqueeze(1).to(args.device)
            # tgt_mag = mag_gt.unsqueeze(1).to(args.device)
            # inp_pha = pha_us.unsqueeze(1).to(args.device)
            # tgt_pha = pha_gt.unsqueeze(1).to(args.device)
            ksp_us = ksp_us.to(args.device)
            # sens = sens.to(args.device)
            mask = mask.to(args.device)
            # img_gt = img_gt.to(args.device)
            img_us = img_us.to(args.device)
            img_gt_np = img_gt_np.unsqueeze(0).to(args.device).float()
            
            # sens = model_sens(ksp_us, mask)
            # img_us =  T.combine_all_coils(img_us.squeeze(0) , sens.squeeze(0)).unsqueeze(0)
            out,_,_ = model(img_us,ksp_us,mask)

        
        if(args.loss == 'SSIM'):
            loss_cmplx_mse = F.mse_loss(out,img_gt_np.cuda())
            loss_cmplx = loss_cmplx_ssim =  ssim_loss(out, img_gt_np,torch.tensor(img_gt_np.max().item()).unsqueeze(0).cuda())
        else:
            loss_cmplx =  loss_cmplx_mse = F.mse_loss(out,img_gt_np.cuda())
            loss_cmplx_ssim =  ssim_loss(out, img_gt_np,torch.tensor(img_gt_np.max().item()).unsqueeze(0).cuda())
          
            
        losses_cmplx.append(loss_cmplx.item())

        # wandb.log({"Dev_Loss_cmplx_ssim": loss_cmplx_ssim.item(), "Dev_Loss_cmplx_mse": loss_cmplx_mse.item()})

        writer.add_scalar('Dev_Loss_cmplx_ssim',loss_cmplx_ssim, epoch)
        writer.add_scalar('Dev_Loss_cmplx_mse',loss_cmplx_mse, epoch)
        
    return  np.mean(losses_cmplx) , time.perf_counter() - start



def visualize(args, epoch,  model, data_loader, writer):
    def save_image(image,tag):
        image -= image.min()
        image /= image.max()
        # image = image[0,0,:,:].cpu().numpy()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        # print("image",image.shape)
        # return image


    model.eval()
    
    with torch.no_grad():

        for iter, data in enumerate(data_loader):
            # ksp_us/img_us_sens.max(), ksp_t/img_us_sens.max() ,  img_us_sens/img_us_sens.max(), img_gt_sens/img_us_sens.max() , img_us_np/img_us_np.max() , img_gt_np/img_us_np.max() , sens_t , mask ,img_us_sens.max(),img_us_np.max(),fname

            img_list=[]

            ksp_us,img_us,img_us_np,img_gt_np,sens,mask,maxi,fname = data
            
            # inp_mag = mag_us.unsqueeze(1).to(args.device)
            # tgt_mag = mag_gt.unsqueeze(1).to(args.device)
            # inp_pha = pha_us.unsqueeze(1).to(args.device)
            # tgt_pha = pha_gt.unsqueeze(1).to(args.device)
            ksp_us = ksp_us.to(args.device)
            sens_ori = sens.to(args.device)
            mask = mask.to(args.device)
            # img_gt = img_gt.to(args.device)
            img_us = img_us.to(args.device)
            img_gt_np = img_gt_np.unsqueeze(0).to(args.device).float()
            
            out,out_stack,sens= model(img_us,ksp_us,mask)
            # print("sens",sens.shape)
            # img_us_sens =  T.combine_all_coils(img_us.squeeze(0) , sens.squeeze(0)).unsqueeze(0)
            # out,out_stack = model_vs(img_us_sens,ksp_us,sens,mask)
            # print("img_us",img_us.shape)

            # print("shape",out_stack[0].shape,out_stack[1].shape,out_stack[1].shape)

            

            img_us_cmplx_abs = (torch.sqrt(img_us[0,:,:,:,0]**2 + img_us[0,:,:,:,1]**2)).unsqueeze(1).to(args.device)
            # print("img_us_cmplx_abs",img_us_cmplx_abs.shape)
            # out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            
            error_cmplx = torch.abs(out.cuda() - img_gt_np.cuda())
            # print("error_complex",error_cmplx.shape)
            # error_cmplx_abs = (torch.sqrt(error_cmplx[:,:,:,0]**2 + error_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            sens = sens.squeeze(0)
            sens_cmplx_abs = (torch.sqrt(sens[:,:,:,0]**2 + sens[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            
            sens_ori = sens_ori.squeeze(0)
            sens_cmplx_abs_ori = (torch.sqrt(sens_ori[:,:,:,0]**2 + sens_ori[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            
            # print("sens_cmplx_abs",sens_cmplx_abs.shape)
            
           
            # out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            # error_cmplx_abs  = T.pad(error_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device) 
            # img_gt_cmplx_abs  = T.pad(img_gt_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)
            # sens_cmplx_abs  = (img_gt_cmplx_abs[0,0,:,:]).unsqueeze(0).unsqueeze(1).to(args.device) 

            
            save_image(error_cmplx,'Error')
            out = save_image(out, 'Recons')
            # print("img_us",img_us_np.shape)
            # wandb.log({"img": [wandb.Image(norm_image(img_us_np.unsqueeze(0)), caption="US-Image"), wandb.Image(norm_image(out), caption="Reconstruction"),
            # wandb.Image(norm_image(img_gt_np.unsqueeze(0)), caption="GT-Image"),wandb.Image(norm_image(error_cmplx), caption="Error")]})
            save_image(img_gt_np,'Target')
            
            save_image(img_us_cmplx_abs,'US-image')
            save_image(sens_cmplx_abs,'sens')
            save_image(sens_cmplx_abs_ori,'sens_map_from_enlive')


            # wb_save_image(img_us_cmplx_abs,out,img_gt_np,error_cmplx)



            
            
            out_cmplx=out_stack[0]
            out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            # out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            save_image(out_cmplx_abs, 'Recons0')
            
            out_cmplx=out_stack[1]
            out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            # out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            save_image(out_cmplx_abs, 'Recons1')
            
            out_cmplx=out_stack[2]
            out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            # out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            save_image(out_cmplx_abs, 'Recons2')
            
            out_cmplx=out_stack[3]
            out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            # out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            save_image(out_cmplx_abs, 'Recons3')
            
            out_cmplx=out_stack[4]
            out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            # out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            save_image(out_cmplx_abs, 'Recons4')
            

            break



def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss_cmplx,is_new_best_cmplx):

    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model':model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'best_dev_loss_cmplx': best_dev_loss_cmplx,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best_cmplx:   
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')



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
    model = build_model(args)
    if args.data_parallel:

        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(model.parameters(),args.lr,args.weight_decay)

    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    return checkpoint,  model ,   optimizer


def build_optim(params,lr,weight_decay):
    optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(params, lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'vs_summary')

    if args.resume == 'True':

        checkpoint, model,optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss_cmplx = checkpoint['best_dev_loss_cmplx']
        start_epoch = checkpoint['epoch']

        del checkpoint
    else:
        
        model  = build_model(args).to(args.device)
        # wandb.watch(model_vs)
        # wandb.watch(model_sens)

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        optimizer = build_optim(model.parameters(),args.lr,args.weight_decay)
        best_dev_loss_cmplx = 1e9
        start_epoch = 0
        
    logging.info(args)

    logging.info(model)

    
    print("training VarNet from SCRATCH with 12-channels data, using SSIM loss")

    train_loader, dev_loader , display_loader = create_data_loaders(args,args.data_path)  
    print("dataloaders for 12 channels data readdy")  

    scheduler_vs = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    print("Parameters in Model=",T.count_parameters(model)/1000,"K")


    for epoch in range(start_epoch, args.num_epochs):




        train_loss_cmplx, train_time = train_epoch(args, epoch,  model ,  train_loader, optimizer, writer)
        dev_loss_cmplx , dev_time = evaluate(args, epoch, model ,  dev_loader, writer)
        visualize(args, epoch, model ,  display_loader, writer)

        scheduler_vs.step(epoch)
        
        is_new_best_cmplx = dev_loss_cmplx < best_dev_loss_cmplx

        best_dev_loss_cmplx = min(best_dev_loss_cmplx, dev_loss_cmplx)
                
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss_cmplx ,is_new_best_cmplx)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}]  TrainLoss_cmplx = {train_loss_cmplx:.4g}  '
            f'DevLoss_cmplx = {dev_loss_cmplx:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
        # torch.save(model_vs.state_dict(), os.path.join(wandb.run.dir, 'model_vs.pt'))
        # torch.save(model_sens.state_dict(), os.path.join(wandb.run.dir, 'model_sens.pt'))
    writer.close()
    # wandb.finish()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=5000, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', type=str, default='False',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    
    parser.add_argument('--data-path', type=pathlib.Path,help='Path to the dataset')
    
    parser.add_argument('--checkpoint', type=str,help='Path to an existing checkpoint. Used along with "--resume"')

    parser.add_argument('--loss', type=str, default='SSIM')
    parser.add_argument('--residual', type=str, default='False')
    parser.add_argument('--acceleration', type=int,help='Ratio of k-space columns to be sampled. 5x or 10x masks provided')
    parser.add_argument('--dropout', type=float,help='% of nodes in decoder to be dropped')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    main(args)
