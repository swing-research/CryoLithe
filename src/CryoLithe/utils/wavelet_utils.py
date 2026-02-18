import torch
import ptwt


def wavelet_decomposition(volume, wavelet):
    # initial decomposition

    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0)

    volume_wt = ptwt.wavedec3(volume, wavelet=wavelet, level=1)

    # Convert to batch mode

    volume_wt_set = [volume_wt[0]]



    keys = list(volume_wt[1].keys())

    for key in keys:
        volume_wt_set.append(volume_wt[1][key])

    #convert to batch mode
    volume_wt_st = torch.concatenate(volume_wt_set, dim=0)
    return volume_wt_st


def wavelet_multilevel_decomposition(volume, wavelet ,levels):

    for i in range(levels):
        volume = wavelet_decomposition(volume, wavelet)
    return volume


def wavelet_reconstruction(volume_wt, wavelet):
    # Reconstruct a batch of wavelet coefficients
    # Bx8xn1xn2xn3

    keys = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']

    vol_wt_set = [volume_wt[:,0]]


    vol_wt_dict = {}
    for i, key in enumerate(keys):
        vol_wt_dict[key] = volume_wt[:,i+1]

    vol_wt_set.append(vol_wt_dict)

    vol_reconstructed = ptwt.waverec3(vol_wt_set, wavelet=wavelet)

    return vol_reconstructed


def wavelet_multilevel_reconstruction(volume_wt, wavelet):

    # Reconstruct the volume from the wavelet coefficients
    #BXN1XN2XN3

    B,N1,N2,N3 = volume_wt.shape
    while B>1:
        volume_wt = volume_wt.view(B//8,8,N1,N2,N3)
        if volume_wt.shape[0] >1:
            volume_wt = volume_wt.permute(1,0,2,3,4)
        volume_wt = wavelet_reconstruction(volume_wt, wavelet)
        B,N1,N2,N3 = volume_wt.shape
    return volume_wt.squeeze(0)