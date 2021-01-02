import torch
import torch.nn as nn
import pywt
# from pytorch_wavelets.pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse
from . import utils


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): whether to do the filtering separably or not (the
            naive implementation can be faster on a gpu).
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh)
                coefficients. yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.g0_col = nn.Parameter(filts[0], requires_grad=False)
        self.g1_col = nn.Parameter(filts[1], requires_grad=False)
        self.g0_row = nn.Parameter(filts[2], requires_grad=False)
        self.g1_row = nn.Parameter(filts[3], requires_grad=False)
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = lowlevel.SFB2D.apply(
                ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll


def data_consistency(k, k0, mask, lmbda=0):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    mask = mask.unsqueeze(-1)
    mask = mask.expand_as(k)

    return (1 - mask) * k + mask * (lmbda*k + k0) / (1 + lmbda)

def get_tv(x):
    x = x.float()
    tv_x = torch.sum(((x[:, 0, :, :-1] - x[:, 0, :, 1:])**2 + (x[:, 1, :, :-1] - x[:, 1, :, 1:])**2)**0.5)
    tv_y = torch.sum(((x[:, 0, :-1, :] - x[:, 0, 1:, :])**2 + (x[:, 1, :-1, :] - x[:, 1, 1:, :])**2)**0.5)

    return tv_x + tv_y

def get_wavelets(x, device):
    xfm = DWTForward(J=3, mode='zero', wave='db4').to(device) # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(x)

    batch_size = x.shape[0]
    channels = x.shape[1]
    rows = nextPowerOf2(Yh[0].shape[-2]*2)
    cols = nextPowerOf2(Yh[0].shape[-1]*2)
    wavelets = torch.zeros(batch_size, channels, rows, cols).to(device)
    # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
    for i, band in enumerate(Yh):
        irow = rows // 2**(i+1)
        icol = cols // 2**(i+1)
        wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

    wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients
    return wavelets

def nextPowerOf2(n):
    count = 0;

    # First n in the below  
    # condition is for the  
    # case where n is 0 
    if (n and not(n & (n - 1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;

def loss_with_reg(z, x, w_coeff, tv_coeff, lmbda, device):
    '''
    z is learned variable, output of model
    x is proximity variable
    '''
    l1 = torch.nn.L1Loss(reduction='sum')
    l2 = torch.nn.MSELoss(reduction='sum')
    dc = lmbda*l2(z, x)

    # Regularization
    z = z.permute(0, 3, 1, 2)
    tv = get_tv(z)
    wavelets = get_wavelets(z, device)
    l1_wavelet = l1(wavelets, torch.zeros_like(wavelets)) # we want L1 value by itself, not the error

    reg = w_coeff*l1_wavelet + tv_coeff*tv

    loss = dc + reg

    return loss, dc, reg

def final_loss(x_hat, y, mask, device, reg_only=False):
    w_coeff, tv_coeff = utils.get_reg_coeff()

    l1 = torch.nn.L1Loss(reduction='sum')
    l2 = torch.nn.MSELoss(reduction='sum')
 
    mask_expand = mask.unsqueeze(2)
 
    # Data consistency term
    Fx_hat = utils.fft(x_hat)
    UFx_hat = Fx_hat * mask_expand
    dc = l2(UFx_hat, y)

    # Regularization
    x_hat = x_hat.permute(0, 3, 1, 2)
    tv = get_tv(x_hat)
    wavelets = get_wavelets(x_hat, device)
    l1_wavelet = l1(wavelets, torch.zeros_like(wavelets)) # we want L1 value by itself, not the error
 
    reg = w_coeff*l1_wavelet + tv_coeff*tv

    if reg_only:
        loss = reg
    else:
        loss = dc + reg

    return loss
"""
    x_hat = complex reconstructions
    gt = ground truth image
    y = US kspace
    mask = US mask
    device = CPU/GPU
    strategy=US or supervised
    
    
"""


def get_loss(x_hat, gt, y, mask, device, strategy, batch_idx, epoch, phase, max_batch):  
    print('strategy',strategy) 
    if strategy == 'unsup':
        loss = final_loss(x_hat, y, mask, device)
    elif strategy == 'sup':
        loss = nn.MSELoss()(x_hat, gt)

    # Trains supervised for 20 epochs, and only on first 1/10 of the dataset
    # Trains unsupervised for rest of the epochs, and on 9/10 of the dataset
    elif strategy == 'refine':
        if phase == 'train' and epoch <= 100:
            if batch_idx < max_batch // 2:
                loss = nn.MSELoss()(x_hat, gt)
                print('sup on batch_idx', str(batch_idx))
            else:
                loss = None
                print('sup skipping batch_idx', str(batch_idx))
        elif phase == 'train' and epoch > 100:
            if batch_idx >= max_batch // 2:
                loss = final_loss(x_hat, y, mask, device)
                print('unsup on batch_idx', str(batch_idx))
            else:
                loss = None
                print('unsup skipping batch_idx', str(batch_idx))
        else:
            loss = nn.MSELoss()(x_hat, gt)
            print('validation in refine')

    return loss