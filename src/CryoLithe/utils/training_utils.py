import torch
from .utils import generate_patches_from_volume_location
from typing import List, Union

#TODO: may be move it to the config file
ALIGN_CORNERS = True   

def generate_patches(vol: Union[torch.Tensor, List[torch.Tensor]], 
                     projections: Union[torch.Tensor, List[torch.Tensor]], 
                     angles: torch.Tensor, patch_size: int, 
                     n_points: int, boundary : bool = False,
                     discrete_sampling: bool = False,
                     patch_scale: torch.Tensor = 1,
                     volume_patch_size: int = None,
                     nlims: int = None,
                    scaled_patches = False):
    """
    using the projections and volume generate the volumme sample and the patches from the projections
    which are generated from the volume sample
     (keep it false)boundary: if true the patches are generated 
            from the boundary of the volume TODO: implement this
    Note: usefull for training the network
    TODO: Probably take only list of volumes and projections
    """


   

    if type(projections) == list:
        n_projections = projections[0].shape[0]
        projection_size = projections[0].shape[1]
        device = projections[0].device
        dtype = projections[0].dtype
    else:
        n_projections = projections.shape[0]
        projection_size = projections.shape[1]
        device = projections[1].device
        dtype = projections.dtype

    if type(vol) == list:
        vol_size = torch.tensor(vol[0].shape, device= device)
    else:
        vol_size = torch.tensor(vol.shape,device= device)
    vol_max_size = max(vol_size)

    scale = vol_size/vol_max_size


    if discrete_sampling:
        if ALIGN_CORNERS:
            if  type(vol) == list:
                n1 = vol[0].shape[0]
                n2 = vol[0].shape[1]
                n3 = vol[0].shape[2]
            else:
                n1 = vol.shape[0]
                n2 = vol.shape[1]
                n3 = vol.shape[2]

            x_range = torch.linspace(-1,1,n1,device = device,dtype=dtype)
            y_range = torch.linspace(-1,1,n2,device = device,dtype=dtype)
            z_range = torch.linspace(-1,1,n3,device = device,dtype=dtype)

            vol_locations = torch.zeros((n_points,3), device = device,dtype=dtype)
            if nlims is not None:
                vol_locations[:,0] = z_range[torch.randint(nlims[2][0],nlims[2][1],(n_points,))]*scale[2]
                vol_locations[:,1] = y_range[torch.randint(nlims[1][0],nlims[1][1],(n_points,))]*scale[1]
                vol_locations[:,2] = x_range[torch.randint(nlims[0][0],nlims[0][1],(n_points,))]*scale[0]
            else:
                vol_locations[:,0] = z_range[torch.randint(0,n3,(n_points,))]*scale[2]
                vol_locations[:,1] = y_range[torch.randint(0,n2,(n_points,))]*scale[1]
                vol_locations[:,2] = x_range[torch.randint(0,n1,(n_points,))]*scale[0]
        # TODO:  add the boundary condition for discrete sampling
    else:
        vol_locations = torch.rand((n_points,3),device=device,dtype=dtype)*2-1
        vol_locations[:,0] = vol_locations[:,0]*scale[2]
        vol_locations[:,1] = vol_locations[:,1]*scale[1]
        vol_locations[:,2] = vol_locations[:,2]*scale[0]

        # Scale the volumes location for which they go out of the circle
        # if they are outside the circle the projections will all zeros for some of the angles

        vol_loc_2norm = torch.clip(torch.linalg.norm(vol_locations[:,:2], dim=1),min=1)
        vol_locations[:,:2] = vol_locations[:,:2]/vol_loc_2norm[:,None]
        if boundary:
            # we will not sample the patches close to the boundary of the volume. The
            # closeness is determined by using the projection angles and patch size
            if type(vol) == list:
                p_scale = 2*patch_size/vol[0].shape[1]
            else:
                p_scale = 2*patch_size/vol.shape[1] # Note the 2 might not be needed
            #angle_abs_max = torch.abs(angles).max()

            #delta = scale[2]*torch.tan(angle_abs_max) + p_scale
            # vol_locations[:,0]  = vol_locations[:,0]
            # vol_locations[:,1]  = vol_locations[:,1]*(1 - torch.clip((abs(vol_locations[:,0])*torch.tan(angle_abs_max) + p_scale+delta),max=1))
            # vol_locations[:,2]  = vol_locations[:,2]*(1 -p_scale)

            vol_locations[:,0]  = vol_locations[:,0]
            vol_locations[:,1]  = vol_locations[:,1]*(1 - p_scale)
            vol_locations[:,2]  = vol_locations[:,2]*(1 -p_scale)

    vol_samples, patches = generate_patches_from_volume_location(vol_locations,vol,
                                                                 projections,
                                                                 angles,
                                                                 patch_size,
                                                                 scale,
                                                                 patch_scale = patch_scale,
                                                                 discrete_sampling = discrete_sampling,
                                                                 scaled_patches = scaled_patches)

    if volume_patch_size is not None:
        volume_patch_samples = volume_sample_patch(vol,vol_locations,patch_size=volume_patch_size,scale = scale)
        return vol_samples,patches,volume_patch_samples

    return vol_samples,patches 

def volume_sample_patch(volume,
                        volume_location,
                        patch_size = 10,
                        scale: torch.Tensor = None,):
    '''
    
    works only for align corners true for now
    '''

    device = volume_location.device
    N_samples = volume_location.shape[0]

    if type(volume) == list:
        n_1,n_2,n_3  = volume[0].shape
    else:
        n_1,n_2,n_3  = volume.shape
    scaled_vol_location = torch.zeros_like(volume_location)

    if scale is None:
        scaled_vol_location[:,0] = volume_location[:,0]
        scaled_vol_location[:,1] = volume_location[:,1]
        scaled_vol_location[:,2] = volume_location[:,2]
    else: 
        scaled_vol_location[:,0] = volume_location[:,0]/scale[2]
        scaled_vol_location[:,1] = volume_location[:,1]/scale[1]
        scaled_vol_location[:,2] = volume_location[:,2]/scale[0]

    

    
    x_patch = (torch.linspace(-1,1,n_2,device=device)[:patch_size])
    x_patch = x_patch - x_patch[patch_size//2]
    y_patch = (torch.linspace(-1,1,n_1,device=device)[:patch_size])
    y_patch = y_patch - y_patch[patch_size//2]
    z_patch = (torch.linspace(-1,1,n_3,device=device)[:patch_size])
    z_patch = z_patch - z_patch[patch_size//2]
    
    xx_pathc, yy_pathc,zz_pathc = torch.meshgrid(y_patch, x_patch, z_patch, indexing='ij')
    points_patch = torch.cat([zz_pathc.unsqueeze(-1),yy_pathc.unsqueeze(-1),xx_pathc.unsqueeze(-1)],dim= -1)
    points_patch = points_patch.reshape(-1,3)


    vol_locations = scaled_vol_location[:,None] + points_patch[None]

    if type(volume) == list:
        vol_samples= []

        for vol in volume:
            vol_samples.append(torch.nn.functional.grid_sample(vol.unsqueeze(0).unsqueeze(0),
                                    vol_locations[None,None],
                                    mode='bilinear',
                                    padding_mode='zeros', 
                                    align_corners=ALIGN_CORNERS).squeeze().reshape(N_samples,patch_size,patch_size,patch_size))
        return vol_samples
    else:
        return torch.nn.functional.grid_sample(volume.unsqueeze(0).unsqueeze(0),
                                        vol_locations[None,None],
                                        mode='bilinear',
                                        padding_mode='zeros', 
                                        align_corners=ALIGN_CORNERS).squeeze().reshape(N_samples,patch_size,patch_size,patch_size)
    