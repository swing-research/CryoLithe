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
        #self.projections = projections
        #self.angles = angles
        #self.patch_scale = patch_scale
        self.scale = scale
        self.configs = configs

        self.register_buffer("projections", projections)
        self.register_buffer("angles", angles)
        self.register_buffer("volume_dummy", volume_dummy)
        self.register_buffer("patch_scale", patch_scale)

        #self.register_buffer("volume_dummy", torch.zeros(100, 100, 100))

    def forward(self, points):


        _, projection_patches = generate_patches_from_volume_location(points, self.volume_dummy ,
                                                                            self.projections,
                                                                            self.angles,
                                                                            patch_size = self.configs.model.patch_size,
                                                                        scale=self.scale,
                                                                        patch_scale= self.patch_scale)
        vol_est = self.model(projection_patches)
        return vol_est