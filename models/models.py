import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import itertools
from data.transforms import ifft2,fft2,fftshift
import data.transforms as T
import math
# from dropout import ConcreteDropout
'''
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,residual='False'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.residual=residual
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        print("U-Net")
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
        if self.residual=='True':
            return self.conv2(output)+input
        else:
            return self.conv2(output)
        
'''           

class DataConsistencyLayer(nn.Module):

    def __init__(self,mask_path,acc_factor,device):
        
        super(DataConsistencyLayer,self).__init__()

        # print (mask_path)
        mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        self.mask = torch.from_numpy(np.load(mask_path)).unsqueeze(2).unsqueeze(0).to(device)

    def forward(self,us_kspace,predicted_img):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        # print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.mask.shape)
        
        updated_kspace1  = self.mask * us_kspace 
        updated_kspace2  = (1 - self.mask) * kspace_predicted_img

        

        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()

    

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        #res1
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        #res4
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        #res5
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out

class Recon_Block(nn.Module):
    def __init__(self):
        super(Recon_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output
    
## Pro version of DUNet   
    
class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()   ##it was PRelu
        self.conv_down = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.recursive_A = _Residual_Block()
        self.recursive_B = _Residual_Block()
        self.recursive_C = _Residual_Block()
        # self.recursive_D = _Residual_Block()
        # self.recursive_E = _Residual_Block()
        # self.recursive_F = _Residual_Block()

        self.recon = Recon_Block()
        #concat

        # self.conv_mid = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mid = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):

        residual = x
        out = self.relu1(self.conv_input(x))
        # print("out1",out.max())
        out = self.relu2(self.conv_down(out))
        # print("out2",out.max())

        out1 = self.recursive_A(out)
        out2 = self.recursive_B(out1)
        out3 = self.recursive_C(out2)
        # out4 = self.recursive_D(out3)
        # out5 = self.recursive_E(out4)
        # out6 = self.recursive_F(out5)

        recon1 = self.recon(out1)
        recon2 = self.recon(out2)
        recon3 = self.recon(out3)
        # recon4 = self.recon(out4)
        # recon5 = self.recon(out5)
        # recon6 = self.recon(out6)
        
        out = torch.cat([recon1, recon2, recon3], 1) 

        # out = torch.cat([recon1, recon2, recon3, recon4, recon5, recon6], 1)
        # out = torch.cat([recon1], 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)
        out = torch.add(out, residual)

        return out


## DUNet lite version 

class _NetG_lite(nn.Module):
    def __init__(self):
        super(_NetG_lite, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()   ##it was PRelu
        self.conv_down = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.recursive_A = _Residual_Block()
#         self.recursive_B = _Residual_Block()
#         self.recursive_C = _Residual_Block()
#         self.recursive_D = _Residual_Block()
#         self.recursive_E = _Residual_Block()
#         self.recursive_F = _Residual_Block()

        self.recon = Recon_Block()
        #concat

        # self.conv_mid = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):
        # print("x",x.max())
        residual = x
        out = self.relu1(self.conv_input(x))
        # print("out1",out.max())
        out = self.relu2(self.conv_down(out))
        # print("out2",out.max())

        out1 = self.recursive_A(out)
#         out2 = self.recursive_B(out1)
#         out3 = self.recursive_C(out2)
#         out4 = self.recursive_D(out3)
#         out5 = self.recursive_E(out4)
#         out6 = self.recursive_F(out5)

        recon1 = self.recon(out1)
#         recon2 = self.recon(out2)
#         recon3 = self.recon(out3)
#         recon4 = self.recon(out4)
#         recon5 = self.recon(out5)
#         recon6 = self.recon(out6)

#         out = torch.cat([recon1, recon2, recon3, recon4, recon5, recon6], 1)
        out = torch.cat([recon1], 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)
        out = torch.add(out, residual)

        return out
    
## VS-Net architecture
    
class dataConsistencyTerm(nn.Module):
    
    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))

    def perform(self,out_img_cmplx,ksp,sens,mask ):
        
        x = T.complex_multiply(out_img_cmplx[...,0].unsqueeze(1), out_img_cmplx[...,1].unsqueeze(1), sens[...,0], sens[...,1])
    
        k = (torch.fft(x, 2, normalized=True)).squeeze(1)
        k_shift = T.ifftshift(k, dim=(-3,-2))

        
        sr = 0.85
        Nz = k_shift.shape[-2] 
        Nz_sampled = int(np.ceil(Nz*sr))
        k_shift[:,:,:,Nz_sampled:,:] = 0
        
        v = self.noise_lvl

        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k_shift + mask * (v * k_shift + (1 - v) * ksp) 
        
        else:
           
            out = (1 - mask) * k_shift + mask * ksp
            
        
        x = torch.ifft(out, 2, normalized=True)
    
        Sx = T.complex_multiply(x[...,0], x[...,1], sens[...,0],-sens[...,1]).sum(dim=1)
        
        Ss = T.complex_multiply(sens[...,0], sens[...,1], sens[...,0],-sens[...,1]).sum(dim=1)
     
        return Sx, Ss
    

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
        return x




class cnn_layer(nn.Module):
    
    def __init__(self):
        super(cnn_layer, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(drop_prob),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(drop_prob),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(drop_prob),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(drop_prob),
            nn.Conv2d(64, 2,  3, padding=1, bias=True),            
        )     
        
    def forward(self, x):
        
        # print("X",x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x
        


    
class network(nn.Module):
    
    def __init__(self, alfa=1, beta=1, cascades=5):
        super(network, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(cnn_layer())  
            dc_blocks.append(dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)

    def forward(self, x, k, m, c):

        op=[]        
        for i in range(self.cascades):

            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
            op.append(x)
 
        img_mag = T.rss(x,m).float()
        return img_mag , op
    

class classifier(nn.Module):
    
    def __init__(self):
        super(classifier, self).__init__()

        self.fc1 = nn.Linear(256*256, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):

        x = T.pad(x.squeeze(0).squeeze(0),[256,256]).unsqueeze(0)
        x =  torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class architecture(nn.Module):

    def __init__(self,dccoeff=0.1,wacoeff=0.1,cascade=5,sens_chans=8, sens_pools=4):
        super(architecture,self).__init__()
        self.dccoeff = dccoeff
        self.wacoeff = wacoeff
        self.cascade = cascade
        self.sens_chans = sens_chans
        self.sens_pools = sens_pools

        self.model_vs = network(self.dccoeff, self.wacoeff, self.cascade)
        self.model_sens = SensitivityModel(self.sens_chans, self.sens_pools)

    def forward(self,img_us,ksp_us,mask):

        sens = self.model_sens(ksp_us, mask)

        img_us =  T.combine_all_coils(img_us.squeeze(0) , sens.squeeze(0)).unsqueeze(0)

        out,outstack = self.model_vs(img_us,ksp_us,sens,mask)

        return out, outstack, sens


    
    
'''    
class network(nn.Module):
    
    def __init__(self, alfa=1, beta=1, cascades=5):
        super(network, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(cnn_layer1()) 
            dc_blocks.append(dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 

        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
        # print(self.conv_blocks)
        # print(self.dc_blocks)
        # print(self.wa_blocks)
 
    def forward(self, x, k, m, c):
        op=[]        
        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
            op.append(x)
        return x , op
'''  


class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(
            1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2,
                          ux ** 2 + uy ** 2 + C1, vx + vy + C2)
        D = B1 * B2
        S = (A1 * A2) / D
        return 1 - S.mean()
   
 ### Dautomap ####
    
def init_noise_(tensor, init):
    with torch.no_grad():
        return getattr(torch.nn.init, init)(tensor) if init else tensor.zero_()


def init_fourier_(tensor, norm='ortho'):
    """Initialise convolution weight with Inverse Fourier Transform"""
    with torch.no_grad():
        # tensor should have shape: (nc_out, nc_in, kx, ky)=(2*N, 2, N, kernel_size)
        nc_out, nc_in, N, kernel_size = tensor.shape

        for k in range(N):
            for n in range(N):
                tensor.data[k, 0, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)
                tensor.data[k, 1, n, kernel_size // 2] = -np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 0, n, kernel_size // 2] = np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 1, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)

        if norm == 'ortho':
            tensor.data[...] = tensor.data[...] / np.sqrt(N)

        return tensor

def get_refinement_block(model='automap_scae', in_channel=1, out_channel=1):
    if model == 'automap_scae':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 5, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, out_channel, 7, 1, 3))
    elif model == 'simple':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, out_channel, 3, 1, 1))
    else:
        raise NotImplementedError


# def init_fourier_2d(N, M, inverse=True, norm='ortho', out_tensor=None,
#                     complex_type=np.complex64):
#     """Initialise fully connected layer as 2D Fourier transform

#     Parameters
#     ----------

#     N, M: a number of rows and columns

#     inverse: bool (default: True) - if True, initialise with the weights for
#     inverse fourier transform

#     norm: 'ortho' or None (default: 'ortho')

#     out_tensor: torch.Tensor (default: None) - if given, copies the values to
#     out_tensor

#     """
#     dft1mat_m = np.zeros((M, M), dtype=complex_type)
#     dft1mat_n = np.zeros((N, N), dtype=complex_type)
#     sign = 1 if inverse else -1

#     for (l, m) in itertools.product(range(M), range(M)):
#         dft1mat_m[l,m] = np.exp(sign * 2 * np.pi * 1j * (m * l / M))

#     for (k, n) in itertools.product(range(N), range(N)):
#         dft1mat_n[k,n] = np.exp(sign * 2 * np.pi * 1j * (n * k / N))

#     # kronecker product
#     mat_kron = np.kron(dft1mat_n, dft1mat_m)

#     # split complex channels into two real channels
#     mat_split = np.block([[np.real(mat_kron), -np.imag(mat_kron)],
#                           [np.imag(mat_kron), np.real(mat_kron)]])

#     if norm == 'ortho':
#         mat_split /= np.sqrt(N * M)
#     elif inverse:
#         mat_split /= (N * M)

#     if out_tensor is not None:
#         out_tensor.data[...] = torch.Tensor(mat_split)
#     else:
#         out_tensor = mat_split
#     return out_tensor



class GeneralisedIFT2Layer(nn.Module):

    def __init__(self, nrow, ncol,
                 nch_in, nch_int=None, nch_out=None,
                 kernel_size=1, nl=None,
                 init_fourier=True, init=None, bias=False, batch_norm=False,
                 share_tfxs=False, learnable=True):
        """Generalised domain transform layer

        The layer can be initialised as Fourier transform if nch_in == nch_int
        == nch_out == 2 and if init_fourier == True.

        It can also be initialised
        as Fourier transform plus noise by setting init_fourier == True and
        init == 'kaiming', for example.

        If nonlinearity nl is used, it is recommended to set bias = True

        One can use this layer as 2D Fourier transform by setting nch_in == nch_int
        == nch_out == 2 and learnable == False


        Parameters
        ----------
        nrow: int - the number of columns of input

        ncol: int - the number of rows of input

        nch_in: int - the number of input channels. One can put real & complex
        here, or put temporal coil channels, temporal frames, multiple
        z-slices, etc..

        nch_int: int - the number of intermediate channel after the transformation
        has been applied for each row. By default, this is the same as the input channel

        nch_out: int - the number of output channels. By default, this is the same as the input channel

        kernel_size: int - kernel size for second axis of 1d transforms

        init_fourier: bool - initialise generalised kernel with inverse fourier transform

        init_noise: str - initialise generalised kernel with standard initialisation. Option: ['kaiming', 'normal']

        nl: ('tanh', 'sigmoid', 'relu', 'lrelu') - add nonlinearity between two transformations. Currently only supports tanh

        bias: bool - add bias for each kernels

        share_tfxs: bool - whether to share two transformations

        learnable: bool

        """
        super(GeneralisedIFT2Layer, self).__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.nch_in = nch_in
        self.nch_int = nch_int
        self.nch_out = nch_out
        self.kernel_size = kernel_size
        self.init_fourier = init_fourier
        self.init = init
        self.nl = nl

        if not self.nch_int:
            self.nch_int = self.nch_in

        if not self.nch_out:
            self.nch_out = self.nch_in

        # Initialise 1D kernels
        idft1 = torch.nn.Conv2d(self.nch_in, self.nch_int * self.nrow, (self.nrow, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)
        idft2 = torch.nn.Conv2d(self.nch_int, self.nch_out * self.ncol, (self.ncol, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)

        # initialise kernels
        init_noise_(idft1.weight, self.init)
        init_noise_(idft2.weight, self.init)

        if self.init_fourier:
            if not (self.nch_in == self.nch_int == self.nch_out == 2):
                raise ValueError

            if self.init:
                # scale the random weights to make it compatible with FFT basis
                idft1.weight.data = F.normalize(idft1.weight.data, dim=2)
                idft2.weight.data = F.normalize(idft2.weight.data, dim=2)

            init_fourier_(idft1.weight)
            init_fourier_(idft2.weight)

        self.idft1 = idft1
        self.idft2 = idft2

        # Allow sharing weights between two transforms if the input size are the same.
        if share_tfxs and nrow == ncol:
            self.idft2 = self.idft1

        self.learnable = learnable
        self.set_learnable(self.learnable)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(self.nch_int)
            self.bn2 = torch.nn.BatchNorm2d(self.nch_out)

    def forward(self, X):
        # input shape should be (batch_size, nc, nx, ny)
        batch_size = len(X)
        # first transform
        x_t = self.idft1(X)

        # reshape & transform
        x_t = x_t.reshape([batch_size, self.nch_int, self.nrow, self.ncol]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn1(x_t.contiguous())

        if self.nl:
            if self.nl == 'tanh':
                x_t = F.tanh(x_t)
            elif self.nl == 'relu':
                x_t = F.relu(x_t)
            elif self.nl == 'sigmoid':
                x_t = F.sigmoid(x_t)
            else:
                raise ValueError

        # second transform
        x_t = self.idft2(x_t)
        x_t = x_t.reshape([batch_size, self.nch_out, self.ncol, self.nrow]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn2(x_t.contiguous())


        return x_t

    def set_learnable(self, flag=True):
        self.learnable = flag
        self.idft1.weight.requires_grad = flag
        self.idft2.weight.requires_grad = flag




# class AUTOMAP(nn.Module):
#     """
#     Pytorch implementation of AUTOMAP [1].

#     Reference:
#     ----------
#     [1] Zhu et al., AUTOMAP, Nature 2018. <url:https://www.nature.com/articles/nature25988.pdf>
#     """

#     def __init__(self, input_shape, output_shape,
#                  init_fc2_fourier=False,
#                  init_fc3_fourier=False):
#         super(AUTOMAP, self).__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.ndim = input_shape[-1]

#         # "Mapped to hidden layer of n^2, activated by tanh"
#         self.input_reshape = int(np.prod(self.input_shape))
#         self.output_reshape = int(np.prod(self.output_shape))

#         self.domain_transform = nn.Linear(self.input_reshape, self.output_reshape)
#         self.domain_transform2 = nn.Linear(self.output_reshape, self.output_reshape)

#         if init_fc2_fourier or init_fc3_fourier:
#             if input_shape != output_shape:
#                 raise ValueError('To initialise the kernels with Fourier transform,'
#                                  'the input and output shapes must be the same')

#         if init_fc2_fourier:
#             init_fourier_2d(input_shape[-2], input_shape[-1], self.domain_transform.weight)

#         if init_fc3_fourier:
#             init_fourier_2d(input_shape[-2], input_shape[-1], self.domain_transform2.weight)


#         # Sparse convolutional autoencoder for further finetuning
#         # See AUTOMAP paper
#         self.sparse_convolutional_autoencoder = get_refinement_block('automap_scae', output_shape[0], output_shape[0])

#     def forward(self, x):
#         """Expects input_shape (batch_size, 2, ndim, ndim)"""
#         batch_size = len(x)
#         x = x.reshape(batch_size, int(np.prod(self.input_shape)))
#         x = F.tanh(self.domain_transform(x))
#         x = F.tanh(self.domain_transform2(x))
#         x = x.reshape(-1, *self.output_shape)
#         x = self.sparse_convolutional_autoencoder(x)
#         return x


class dAUTOMAP(nn.Module):
    """
    Pytorch implementation of dAUTOMAP

    Decomposes the automap kernel into 2 Generalised "1D" transforms to make it scalable.
    """
    def __init__(self, input_shape, output_shape, tfx_params, tfx_params2=None):
        super(dAUTOMAP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        if tfx_params2 is None:
            tfx_params2 = tfx_params

        self.domain_transform = GeneralisedIFT2Layer(**tfx_params)
        # self.domain_transform2 = GeneralisedIFT2Layer(**tfx_params2)
        # self.refinement_block = get_refinement_block('automap_scae', input_shape[0], output_shape[0])

    def forward(self, x):
        """Assumes input to be (batch_size, 2, nrow, ncol)"""
        x_mapped = self.domain_transform(x)
        # x_mapped = F.tanh(x_mapped)
        # x_mapped2 = self.domain_transform2(x_mapped)
        # x_mapped2 = F.tanh(x_mapped2)
        # out = self.refinement_block(x_mapped2)
        return x_mapped


# class dAUTOMAPExt(nn.Module):
#     """
#     Pytorch implementation of dAUTOMAP with adjustable depth and nonlinearity

#     Decomposes the automap kernel into 2 Generalised "1D" transforms to make it scalable.

#     Parameters
#     ----------

#     input_shape: tuple (n_channel, nx, ny)

#     output_shape: tuple (n_channel, nx, ny)

#     depth: int (default: 2)

#     tfx_params: list of dict or dict. If list of dict, it must provide the parameter for each. If dict, then the same parameter config will be shared for all the layers.


#     """
#     def __init__(self, input_shape, output_shape, tfx_params=None, depth=2, nl='tanh'):
#         super(dAUTOMAPExt, self).__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.depth = depth
#         self.nl = nl

#         # copy tfx_parameters
#         domain_transforms = []
#         if isinstance(tfx_params, list):
#             if self.depth and self.depth != len(tfx_params):
#                 raise ValueError('Depth and the length of tfx_params must be the same')
#         else:
#             tfx_params = [tfx_params] * self.depth

#         # create domain transform layers
#         for tfx_param in tfx_params:
#             domain_transform = GeneralisedIFT2Layer(**tfx_param)
#             domain_transforms.append(domain_transform)

#         self.domain_transforms = nn.ModuleList(domain_transforms)
#         self.refinement_block = get_refinement_block('automap_scae', input_shape[0], output_shape[0])

#     def forward(self, x):
#         """Assumes input to be (batch_size, 2, nrow, ncol)"""
#         for i in range(self.depth):
#             x = self.domain_transforms[i](x)
#             x = getattr(F, self.nl)(x)

#         out = self.refinement_block(x)
#         return out

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1 # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1 # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output




class NormUnet(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.unet = UnetModel(
            in_chans=2,
            out_chans=2,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=0
        )

    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def pad(self, x):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        b, c, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x, h_pad, w_pad, h_mult, w_mult):
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, x):
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x

class VarNetBlock(nn.Module):
    def __init__(self, model):
        super(VarNetBlock, self).__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.register_buffer('zero', torch.zeros(1, 1, 1, 1, 1))

    def forward(self, current_kspace, ref_kspace, mask, sens_maps):
        def sens_expand(x):
            return T.fft2(T.complex_mul(x, sens_maps))

        def sens_reduce(x):
            x = T.ifft2(x)
            return T.complex_mul(x, T.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

        def soft_dc(x):
            return torch.where(mask, x - ref_kspace, self.zero) * self.dc_weight

        return current_kspace - \
            soft_dc(current_kspace) - \
            sens_expand(self.model(sens_reduce(current_kspace)))
            
            
class SensitivityModel(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools)

    def chans_to_batch_dim(self, x):
        b, c, *other = x.shape
        return x.contiguous().view(b * c, 1, *other), b

    def batch_chans_to_chan_dim(self, x, batch_size):
        bc, one, *other = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, *other)
    
    

    def divide_root_sum_of_squares(self, x):
        
        def root_sum_of_squares_complex(data, dim=0):
            """
            Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
            Args:
                data (torch.Tensor): The input tensor
                dim (int): The dimensions along which to apply the RSS transform
            Returns:
                torch.Tensor: The RSS value
            """
            return torch.sqrt(complex_abs_sq(data).sum(dim))
        
        def complex_abs_sq(data):
            """
            Compute the squared absolute value of a complex tensor
            """
            assert data.size(-1) == 2
            return (data ** 2).sum(dim=-1)
        return x / root_sum_of_squares_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace, mask):
        
#         def mask_centre(ksp,mask):
#             mask = mask.squeeze(0)
#             x_c = mask.shape[0]//2
#             y_c = mask.shape[1]//2
#             r = 0

#             x_c2 = x_c
#             y_c2 = y_c

#             while (mask[x_c2,y_c2,0]):
# #                 print("r=",r)
#                 r = r+1
#                 x_c2 = x_c2+1
#             r = r+2
#             mask2 = torch.zeros_like(ksp)
#             for x in range(x_c-r,x_c+r):
#                 for y in range(y_c-r,y_c+r):
#                     d = np.sqrt((x-x_c)**2+(y-y_c)**2)
#                     if(d<=r):
# #                         print(x,y)
#                         mask2[:,:,x,y,:] = ksp[:,:,x,y,:] # b,32,218,180,2
#             return mask2


        def mask_centre(ksp,mask):
            # print("mask",mask.shape)
            mask = mask.squeeze(0)
            x_c = mask.shape[0]//2
            y_c = mask.shape[1]//2
            r = 0

            x_c2 = x_c
            y_c2 = y_c

            while (mask[x_c2,y_c2,0]):
        #                 print("r=",r)
                r = r+1
                x_c2 = x_c2+1
        #     r = r+2
            mask2 = torch.zeros_like(ksp)
        #     print(int(x_c-r//torch.sqrt(torch.tensor(2.0))))
            
            for x in range(int(x_c-r/torch.sqrt(torch.tensor(2.0))),int(x_c+r/torch.sqrt(torch.tensor(2.0)))):
                for y in range(int(y_c-r/torch.sqrt(torch.tensor(2.0))),int(y_c+r/torch.sqrt(torch.tensor(2.0)))):
        #             d = np.sqrt((x-x_c)**2+(y-y_c)**2)
        #             if(d<=r):
        #             print(x,y)
                    mask2[:,:,x,y,:] = ksp[:,:,x,y,:] # b,32,218,180,2
            return mask2

        x = mask_centre(masked_kspace,mask)
        x = T.ifft2(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        
        x = self.divide_root_sum_of_squares(x)
        
        return x
            


