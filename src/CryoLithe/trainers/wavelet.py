import torch
import numpy as np
from .real import TrainerReal
from ..utils.utils import  custom_ramp_fft
from ..utils.training_utils import generate_patches
from ..utils.wavelet_utils import wavelet_multilevel_decomposition

class TrainerRealWavelet(TrainerReal):
    """The trainer for the real data  where the model outputs a wavelet coefficient instead of a single point, the loss is computed on a subvolume"""
    def train_step(self,data_loader, device = 'cpu', wandb_run = None):
        """
        Train the model for one epoch
        """
        self.model.train()

        # check if wavelet loss coefficients are present if not add them
        if not hasattr(self,'wavelet_train_loss'):
            self.wavelet_train_loss = []

        train_loss_epoch = []
        wavelet_loss_epoch = []
        for i, train_data in enumerate(data_loader):
            volume = train_data['volume'][0].to(device)
            projection = train_data['projection'][0].to(device)
            angles = train_data['angles'][0].to(device)            
            projection_filt = projection.clone()

            if self.config.training.dynamic_zlims:
                zlim_values = train_data['z_lims']
            else:
                zlim_values = None


            if self.config.use_2D_filters:
                projection_filt = self.filter_2D(projection_filt)
            if self.config.filter_projections:
                projection_fiter = self.ramp(projection_filt.shape[2])
                projection_filt = custom_ramp_fft(projection_filt,projection_fiter)
            
            self.optimizer.zero_grad()
            loss = self.compute_loss(volume,projection_filt,angles,zlims = zlim_values)
            # 
            wavelet_loss_epoch.append(loss.detach().cpu().numpy())
            loss = loss.sum()
            

            loss.backward()
            self.optimizer.step()
            train_loss_epoch.append(loss.item()) 
        self.scheduler.step()
        self.train_loss.append(np.mean(train_loss_epoch)) 
        self.wavelet_train_loss.append(np.mean(wavelet_loss_epoch, axis=0))   
        if wandb_run is not None:
            wavelet_loss = np.mean(wavelet_loss_epoch, axis=0)
            tr_loss = np.mean(train_loss_epoch)
            wandb_run.log({"train_loss": tr_loss, "wavelet_train_loss": wavelet_loss}, step=len(self.train_loss))


    def validate(self, valid_loader, device = 'cpu', wandb_run = None):
        """
        valid loader contains the filtered projections
        """

        # check if wavelet loss coefficients are present if not add them
        if not hasattr(self,'wavelet_valid_loss'):
            self.wavelet_valid_loss = []

        with torch.no_grad():
            valid_loss_epc = []
            wavelet_loss_epc = []
            self.model.eval()
            for j, valid_data in enumerate(valid_loader):
                volume = valid_data['volume'][0].to(device)
                projection = valid_data['projection'][0].to(device)
                angles = valid_data['angles'][0].to(device)            
                projection_filt = projection.clone()

                if self.config.training.dynamic_zlims:
                    zlim_values = valid_data['z_lims']
                else:
                    zlim_values = None
                if self.config.use_2D_filters:
                        projection_filt = self.filter_2D(projection_filt)
                if self.config.filter_projections:
                    projection_fiter = self.ramp(projection_filt.shape[2])
                    projection_filt = custom_ramp_fft(projection_filt,projection_fiter) 


                loss = self.compute_loss(volume,projection_filt,angles,zlims = zlim_values)
                wavelet_loss_epc.append(loss.detach().cpu().numpy())
                loss = loss.sum()

                #print('valid loss')
                #print(loss)
                valid_loss_epc.append(loss.item())
            self.valid_loss.append(np.mean(valid_loss_epc))
            self.wavelet_valid_loss.append(np.mean(wavelet_loss_epc, axis=0)) 
            if wandb_run is not None:
                wavelet_loss = np.mean(wavelet_loss_epc, axis=0)
                val_loss = np.mean(valid_loss_epc)
                wandb_run.log({"valid_loss": val_loss, "wavelet_valid_loss": wavelet_loss}, step=len(self.valid_loss)) 


    def save_checkpoint(self, epoch, save_path):
        """
        Save the model checkpoint
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'ramp': self.ramp,
            'patch_scale': self.patch_scale,
            'wavelet_train_loss': self.wavelet_train_loss,
            'wavelet_valid_loss': self.wavelet_valid_loss,
            'random_state': torch.get_rng_state()
        }, save_path)
    
    
    
    def compute_loss(self,volume,projection, angles, zlims = None):
        """
        Compute the loss using the model and criterion for the given volume, projection and angles
        """




        volume = self.compute_wavelet_coefficients(volume)

        if self.config.training.dynamic_zlims:
            zlim_values = zlims
        else:
            zlim_values = self.config.training.nlims


        # If volume size is smaller than zlim values, then adjust the zlim values
        # For now check only the last axiscompute_wavelet_coefficients
        if volume[0].shape[-1] < zlim_values[-1][1]:
            # usse half the size of the volume
            zlims_list = list(zlim_values)
            zlims_list_z = list(zlims_list[-1])
            zlims_list_z[0] = volume[0].shape[-1]//4
            zlims_list_z[1] = 3*volume[0].shape[-1]//4
            zlims_list[-1] = tuple(zlims_list_z)
            zlim_values = tuple(zlims_list)


        vol_samples,patches  = generate_patches(volume,
                                                        projection,
                                                        angles,
                                                        patch_size = self.config.model.patch_size, 
                                                        n_points = self.config.training.num_points,
                                                        patch_scale = self.patch_scale,
                                                        boundary = self.config.training.boundary,
                                                        discrete_sampling = self.config.training.discrete_sampling,
                                                        nlims= zlim_values,
                                                        volume_patch_size = None)
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
            vol_est = self.model(patches)


        # conert volsamples list to tensor
        vol_samples = torch.stack(vol_samples,dim=1)

        
        #print(vol_est.shape, volume_patch_samples.shape)

        # Compute loss for the individual wavelet coefficients


        

        loss = self.criterion(vol_est, vol_samples).mean(0)

        return loss  

    # def compute_wavelet_coefficients(self,volume):
    #     """
    #     Compute the wavelet coefficients of the volume
    #     """

    #     vol_wavelet = ptwt.wavedec3(volume, self.config.training.wavelet, level = 1)
    #     # covert to list
    #     vol_lp = vol_wavelet[0]
    #     vol_hp = list(vol_wavelet[1].values())

    #     vol_list = [vol_lp[0]]
    #     # Removeing the first element of the list
    #     for i in range(len(vol_hp)):
    #         vol_list.append(vol_hp[i].squeeze(0))

    #     return vol_list

    def compute_wavelet_coefficients(self,volume):
        """
        Compute the wavelet coefficients of the volume
        """

        vol_wavelet = wavelet_multilevel_decomposition(volume = volume,
                                                       wavelet = self.config.training.wavelet,
                                                       levels = self.config.training.wavelet_levels)


        # convert to list
        vol_list = list(torch.unbind(vol_wavelet, dim = 0))

        return vol_list
