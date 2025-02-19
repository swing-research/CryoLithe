"""
Script containing utility functions
"""

import torch
import numpy as np
from typing import List
import torch.nn.functional as F
from typing import List, Union


ALIGN_CORNERS = True   


def generate_projections_location(points: torch.FloatTensor, angles: torch.FloatTensor):
    """
    Generate the points in the local coordinates of a plane passing through the origin
    tilted by angle along y axis
    points: (n_points,3) tensor of points
    angles: (n_projections) tensor of angles

    Note: the projections is only for single axis tilts along the y axis
    tilts between (-pi/2,pi/2) are valid pi/2 is not valid
    """

    n_projections = len(angles)
    n_points = len(points)

    normals = torch.zeros((n_projections,3)).to(points.device)

    normals[:,0] = torch.cos(angles)
    normals[:,1] = -torch.sin(angles)

    projection =  points@ normals.T
    points_proj = points[:,None,:] - projection[:,:,None] * normals[None,:,:]
    points_proj[:,:,1] = points_proj[:,:,1]/torch.cos(angles)[None,:]
    local_coords = points_proj[:,:,1:]

    return local_coords.permute(1,0,2)






def volume_sampler(volume,volume_location):
    """
    Uses the sampling function:
    volume : HxWxD 
    volume_location: (N_points,3)
    outputs = N_points
    """
    # volume_location = volume_location[:,[2,1,0]]
    # return interpolate_grid_3d(volume[None],volume_location/2+0.5,CUBIC_B_SPLINE_MATRIX.to(volume.device)).squeeze()

    
    
    return torch.nn.functional.grid_sample(volume.unsqueeze(0).unsqueeze(0),
                                    volume_location.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                    mode='bilinear',
                                    padding_mode='zeros', 
                                    align_corners=ALIGN_CORNERS).squeeze() 


def image_sampler(image,image_location):
    """
    image: HXW
    img_location: N_points,2
    outputs: N_points
    """
    # image_location = image_location[:,[1,0]]
    # return interpolate_grid_2d(image[None],image_location/2+0.5,CUBIC_B_SPLINE_MATRIX.to(image.device)).squeeze()



    return torch.nn.functional.grid_sample(image.unsqueeze(0).unsqueeze(0),
                                image_location.unsqueeze(0).unsqueeze(0),
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=ALIGN_CORNERS).squeeze()



def generate_patches_from_volume_location(volume_location : torch.Tensor, 
                                          vol:  Union[torch.Tensor, List[torch.Tensor]], 
                                          projections: Union[torch.Tensor, List[torch.Tensor]],
                                          angles:Union[torch.Tensor, List[torch.Tensor]], 
                                          patch_size: int,
                                          scale: torch.Tensor = None,
                                          patch_scale: torch.Tensor = 1,
                                          discrete_sampling: bool = False,
                                          scaled_patches: bool = False):
    
    """
    using the volume location and projections generate the projections patches
    volume_location: (n_points,3) tensor of points between -1 and 1
    Note: This now works for rectangular volumes
    TODO: make vol optional
    

    """

    device = volume_location.device
    
    if type(projections) == list:
        n_1 = projections[0].shape[1]
        n_2 = projections[0].shape[2]
        n_projections = projections[0].shape[0]
    else:
        n_1 = projections.shape[1]
        n_2 = projections.shape[2]
        n_projections = projections.shape[0]
    #projection_size = projections.shape[1]
    n_points = volume_location.shape[0]

    proj_scale = torch.tensor([n_1,n_2],device=device)
    proj_scale = proj_scale/max(proj_scale)
    vol_samples = None


    if ALIGN_CORNERS:
        x_patch = (torch.linspace(-1,1,n_2,device=device)[:patch_size])
        x_patch = x_patch - x_patch[patch_size//2]
        y_patch = (torch.linspace(-1,1,n_1,device=device)[:patch_size])
        y_patch = y_patch - y_patch[patch_size//2]
    else:
        x_patch = (torch.arange(-patch_size//2+1,patch_size//2+1, device=device)*2/n_2)
        y_patch = (torch.arange(-patch_size//2+1,patch_size//2+1,device=device)*2/n_1)

    xx_pathc, yy_pathc = torch.meshgrid(x_patch, y_patch, indexing='xy')
    points_patch = torch.zeros((patch_size*patch_size,2),device=device)
    points_patch[:,0] = xx_pathc.flatten()
    points_patch[:,1] = yy_pathc.flatten()

    
    if type(angles) == list:
        projection_locations = []
        for i,angle_vals in enumerate(angles):
            projection_centers = generate_projections_location(volume_location,angle_vals)
            projection_centers[:,:,0] = projection_centers[:,:,0]/proj_scale[1]
            projection_centers[:,:,1] = projection_centers[:,:,1]/proj_scale[0]
            if scaled_patches:
                # Compute the scaling value
                n_curr = projections[i].shape[-1]
                p_scale = n_2/n_curr
                projection_locations.append(projection_centers.unsqueeze(2) + patch_scale*points_patch.unsqueeze(0).unsqueeze(0)*p_scale)
            else:
                projection_locations.append(projection_centers.unsqueeze(2) + patch_scale*points_patch.unsqueeze(0).unsqueeze(0))
    else:
        projection_centers = generate_projections_location(volume_location,angles)
        projection_centers[:,:,0] = projection_centers[:,:,0]/proj_scale[1]
        projection_centers[:,:,1] = projection_centers[:,:,1]/proj_scale[0]
        # Generate patch coordinates
        #print(projection_centers)
        projection_locations = projection_centers.unsqueeze(2) + patch_scale*points_patch.unsqueeze(0).unsqueeze(0)

   


    if type(projections) == list:

        patches_list = []

        for index, projections_i in enumerate(projections):
            if type(projection_locations) == list:
                n_projections = len(angles[index])
                patches = torch.zeros((n_projections,n_points,patch_size,patch_size),device=device)
                projection_locations_current = projection_locations[index]
            else:
                patches = torch.zeros((n_projections,n_points,patch_size,patch_size),device=device)
                projection_locations_current =projection_locations

            for i in range(n_projections):
                i_patch_points = projection_locations_current[i].reshape(-1,2)
                pp = image_sampler(projections_i[i],i_patch_points)
                # pp = torch.nn.functional.grid_sample(projections_i[i].unsqueeze(0).unsqueeze(0),
                #                                             i_patch_points.unsqueeze(
                # 0).unsqueeze(0),mode='bilinear',padding_mode='zeros',align_corners=ALIGN_CORNERS).squeeze()

                patches[i,:,:,:] = pp.reshape(n_points,patch_size,patch_size)
            patches = patches.permute(1,0,2,3)

            patches_list.append(patches)

        return vol_samples,patches_list
    else:

        patches = torch.zeros((n_projections,n_points,patch_size,patch_size),device=device)

        for i in range(n_projections):
            i_patch_points = projection_locations[i].reshape(-1,2)
            pp = image_sampler(projections[i],i_patch_points)
            # pp = torch.nn.functional.grid_sample(projections[i].unsqueeze(0).unsqueeze(0),
            #                                             i_patch_points.unsqueeze(
            # 0).unsqueeze(0),mode='bilinear',padding_mode='zeros',align_corners=ALIGN_CORNERS).squeeze()

        # if i == 30:
        #     #print(i_patch_points)
        #     #print(projections[i].max())

        #     #print(i_patch_points.shape)
        #     #pts = torch.cat([pp.unsqueeze(1),i_patch_points],dim=1)
        #     #print(pts)

            patches[i,:,:,:] = pp.reshape(n_points,patch_size,patch_size)

        patches = patches.permute(1,0,2,3)
        return vol_samples,patches







def custom_ramp_fft(x,t_cust, use_splits = False):
    """ 
    ramp filtering using torch fft 
    x: (n_projections, N , N) tensor
    t_cust: (2*N) tensor
    use_splits: use the fft for each projections separately so that it can be used in low GPU cards
    """
    if use_splits:
        projection_filtered = []
        for proj in x:
            projection_fft = torch.fft.fftn(proj, dim=(-1), s = t_cust.shape[0])
            projection_fft = projection_fft*t_cust[None,:]
            projection_filtered.append(torch.fft.ifftn(projection_fft, dim=(-1), s =t_cust.shape[0]).real[:,0:x.shape[-1]])
        projection_filtered = torch.stack(projection_filtered,dim=0)
    else:
        projection_fft = torch.fft.fftn(x, dim=(-1), s = t_cust.shape[0])
        projection_fft = projection_fft*t_cust[None,None,:]
        projection_filtered = torch.fft.ifftn(projection_fft, dim=(-1), s = t_cust.shape[0]).real[:,:,0:x.shape[-1]]
    return projection_filtered




def vol_normalize(vol,min_Val,max_Val):
    vol = (vol- np.min(vol))/(np.max(vol)-np.min(vol))
    vol = vol*(max_Val-min_Val)+min_Val
    return vol





def downsample_anti_aliasing(vol_t, scale=0.5):
    # anti-aliasing in the frequency domain
    vol_fft = torch.fft.fftn(vol_t)
    vol_shape = ((torch.tensor(vol_t.shape)//2)*scale).int()
    vol_fft[vol_shape[0]:-vol_shape[0], :, :] = 0
    vol_fft[:, vol_shape[1]:-vol_shape[1], :] = 0
    vol_fft[:, :, vol_shape[2]:-vol_shape[2]] = 0
    vol_filtered = torch.fft.ifftn(vol_fft).real

    del vol_fft

    vol_downsampled = F.interpolate(vol_filtered[None,None], scale_factor=scale, mode='trilinear', align_corners=False)
    return vol_downsampled[0,0]






def normalize_numpy(inp, batchwise=True):
    """
    Normalize the input by substracting the mean and divided by the std.

    INPUT:
        -inp, (B,*n): input numpy array
        -batchwise, bool: if True, normalize each batch elements differently
    """
    if batchwise:
        s = np.std(inp,axis=(1,2),keepdims=True)
        out = (inp - np.mean(inp,axis=0,keepdims=True))/s
    else:
        s = inp.std()
        if s!=0:
            out = (inp - inp.mean())/s
    return out


def normalize_torch(inp, batchwise=True):
    """
    Normalize the input by substracting the mean and divided by the std.

    INPUT:
        -inp, (B,*n): inputdd torch tensor
        -batchwise, bool: if True, normalize each batch elements differently
    """
    if batchwise:
        s = torch.std(inp,dim=(1,2),keepdim=True)
        out = (inp - torch.mean(inp,dim=0,keepdim=True))/s
    else:
        s = inp.std()
        if s!=0:
            out = (inp - inp.mean())/s
    return out

def SNR(x_ref,x):
    dif = np.sum((x_ref-x)**2)
    nref = np.sum(x_ref**2)
    res=10*np.log10((nref+1e-16)/(dif+1e-16))
    return res

def torch_filter(img: torch.Tensor,img_filter: torch.Tensor, padding = 0):
    """
    Filter the image using the filter using conv2d
    img: (N,N) tensor or (B,N,N) tensor (batched image
    img_filter: (Nf,Nf) tensor
    """
    if len(img.shape) == 2:
        return F.conv2d(img.unsqueeze(0).unsqueeze(0),img_filter.unsqueeze(0).unsqueeze(0),padding=padding).squeeze(0).squeeze(0)
    else:
        return F.conv2d(img.unsqueeze(1),img_filter.unsqueeze(0).unsqueeze(0),padding=padding).squeeze()




    