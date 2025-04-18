import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import generate_patches_from_volume_location




class model_wrapper(nn.Module):
    def __init__(self, model, 
                 projections, 
                 angles, 
                 volume_dummy,
                 patch_scale, 
                 scale, 
                 configs):
        super(model_wrapper, self).__init__()

        self.model = model

        self.scale = scale
        self.configs = configs

        self.register_buffer("projections", projections)
        self.register_buffer("angles", angles)
        self.register_buffer("volume_dummy", volume_dummy)
        self.register_buffer("patch_scale", patch_scale)


    def forward(self, points):
        """
        Wrapper for the model so that it can be used with the DataParallel
        points: Bx3 
        """


        _, projection_patches = generate_patches_from_volume_location(points, self.volume_dummy ,
                                                                            self.projections,
                                                                            self.angles,
                                                                            patch_size = self.configs.model.patch_size,
                                                                        scale=self.scale,
                                                                        patch_scale= self.patch_scale)
        
        if self.configs.training.use_angle:
            angle_info = self.angles.unsqueeze(0).repeat(projection_patches.shape[0],1)
            vol_est = self.model(projection_patches.half(),angle_info.half())
        else:
            vol_est = self.model(projection_patches.half())
        return vol_est