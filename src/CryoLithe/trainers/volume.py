""" Volume patch trainer (Not tested)"""
import torch

from ..utils.utils import  custom_ramp_fft
from ..utils.training_utils import generate_patches
from .real import TrainerReal


class TrainerRealVolume(TrainerReal):
    """The trainer for the real data  where the model outputs a subvolume instead of a single point, the loss is computed on a subvolume"""
    def compute_loss(self,volume,projection, angles, zlims = None):
        """
        Compute the loss using the model and criterion for the given volume, projection and angles
        """

        if self.config.training.dynamic_zlims:
            zlim_values = zlims
        else:
            zlim_values = self.config.training.nlims

        vol_patch_size = self.config.training.volume_patch_size

        vol_samples,patches,volume_patch_samples  = generate_patches(volume,
                                                        projection,
                                                        angles,
                                                        patch_size = self.config.model.patch_size, 
                                                        n_points = self.config.training.num_points,
                                                        patch_scale = self.patch_scale,
                                                        boundary = self.config.training.boundary,
                                                        discrete_sampling = self.config.training.discrete_sampling,
                                                        nlims= zlim_values,
                                                        volume_patch_size = vol_patch_size)
        if type(patches) is list:
            patches = torch.concat(patches,1)
        if self.config.training.normalize_patch:
            patches_norm = torch.linalg.vector_norm(patches, 
                                                    dim=(-3,-2,-1), 
                                                    ord = float(self.config.training.patch_ord))
            patches = patches/patches_norm[:,None,None,None]

        if self.config.training.use_half_precision:
            patches = patches.half()
            vol_samples = vol_samples.half()
        if self.config.training.use_angle:
            angle_info = angles.unsqueeze(0).repeat(patches.shape[0],1)
            vol_est = self.model(patches,angle_info)
        else:
            vol_est = self.model(patches).reshape(-1,vol_patch_size,vol_patch_size,vol_patch_size)
        #print(vol_est.shape, volume_patch_samples.shape)
        loss = self.criterion(vol_est, volume_patch_samples)

        return loss
    