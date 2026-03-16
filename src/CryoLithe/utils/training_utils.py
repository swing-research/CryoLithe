import torch
from typing import List, Union
from .utils import image_sampler, volume_sampler, generate_projections_location
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
        device = projections.device
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
    dtype = volume_location.dtype
    
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

    if scale is None:

        if type(vol) == list:
            vol_samples = []
            for vol_i in vol:
                vol_samples.append(volume_sampler(vol_i,volume_location))
                # torch.nn.functional.grid_sample(vol_i.unsqueeze(0).unsqueeze(0),
                #                                             volume_location.unsqueeze(
                # 0).unsqueeze(0).unsqueeze(0),mode='bilinear',padding_mode='zeros', align_corners=ALIGN_CORNERS).squeeze())
        else:
            vol_samples = volume_sampler(vol,volume_location)
            # torch.nn.functional.grid_sample(vol.unsqueeze(0).unsqueeze(0),
            #                                                 volume_location.unsqueeze(
            #     0).unsqueeze(0).unsqueeze(0),mode='bilinear',padding_mode='zeros', align_corners=ALIGN_CORNERS).squeeze()
    else:

        if discrete_sampling:
            if type(vol) == list:
                vol_n1 = vol[0].shape[0]
                vol_n2 = vol[0].shape[1]
                vol_n3 = vol[0].shape[2]
            else:
                vol_n1 = vol.shape[0]
                vol_n2 = vol.shape[1]
                vol_n3 = vol.shape[2]

            x_location = torch.round(((volume_location[:,2]/scale[0])/2 + 0.5)*(vol_n1-1)).long()
            y_location = torch.round(((volume_location[:,1]/scale[1])/2 + 0.5)*(vol_n2-1)).long()
            z_location = torch.round(((volume_location[:,0]/scale[2])/2 + 0.5)*(vol_n3-1)).long()


            #print(x_location)
            if type(vol) == list:
                vol_samples = []
                for vol_i in vol:
                    vol_samples.append(vol_i[x_location,y_location,z_location])
            else:
                vol_samples = vol[x_location,y_location,z_location]

        else:
            scaled_vol_location = volume_location.clone()
            scaled_vol_location[:,0] = scaled_vol_location[:,0]/scale[2]
            scaled_vol_location[:,1] = scaled_vol_location[:,1]/scale[1]
            scaled_vol_location[:,2] = scaled_vol_location[:,2]/scale[0]

            if type(vol) == list:
                vol_samples = []
                for vol_i in vol:
                    vol_samples.append(volume_sampler(vol_i,scaled_vol_location))                
            else: 
                vol_samples = volume_sampler(vol,scaled_vol_location)


    proj_scale = torch.tensor([n_1,n_2],device=device,dtype=dtype)
    proj_scale = proj_scale/max(proj_scale)


    if ALIGN_CORNERS:
        x_patch = (torch.linspace(-1,1,n_2,device=device,dtype=dtype)[:patch_size])
        x_patch = x_patch - x_patch[patch_size//2]
        y_patch = (torch.linspace(-1,1,n_1,device=device,dtype=dtype)[:patch_size])
        y_patch = y_patch - y_patch[patch_size//2]
    else:
        x_patch = (torch.arange(-patch_size//2+1,patch_size//2+1, device=device,dtype=dtype)*2/n_2)
        y_patch = (torch.arange(-patch_size//2+1,patch_size//2+1,device=device,dtype=dtype)*2/n_1)

    xx_pathc, yy_pathc = torch.meshgrid(x_patch, y_patch, indexing='xy')
    points_patch = torch.zeros((patch_size*patch_size,2),device=device,dtype=dtype)
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
                patches = torch.zeros((n_projections,n_points,patch_size,patch_size),device=device,dtype=dtype)
                projection_locations_current = projection_locations[index]
            else:
                patches = torch.zeros((n_projections,n_points,patch_size,patch_size),device=device,dtype=dtype)
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

        patches = torch.zeros((n_projections,n_points,patch_size,patch_size),device=device,dtype=dtype)

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
    