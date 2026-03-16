import torch
import numpy as np
from ..utils.utils import  custom_ramp_fft
from ..utils.training_utils import generate_patches


class Trainer:
    def __init__(self, config, 
                 model, 
                 ramp,
                 patch_scale,
                 filter_2D,
                 optimizer, 
                 criterion, 
                 scheduler, 
                 train_loss = [],
                 valid_loss = [],
                 device = 'cpu',
                 dtype = torch.float32):

        self.config = config
        self.model = model
        self.ramp = ramp
        self.patch_scale = patch_scale
        self.filter_2D = filter_2D
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.dtype = dtype

    def load_checkpoint(self, checkpoint_path):
        """
        Load the checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.valid_loss = checkpoint['valid_loss']
        self.ramp = checkpoint['ramp']
        self.patch_scale = checkpoint['patch_scale']
        torch.set_rng_state(checkpoint['random_state'])
        self.start = checkpoint['epoch']+1
        print('Checkpoint loaded')
        return self.start

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
            'random_state': torch.get_rng_state()
        }, save_path)


    def compute_loss(self,volume,projection, angles, zlims = None):
        """
        Compute the loss using the model and criterion for the given volume, projection and angles
        """

        if self.config.training.dynamic_zlims:
            zlim_values = zlims
        else:
            zlim_values = self.config.training.nlims

        # using mixed training
        #print(projection.dtype)
        #print(volume.dtype)
        #print(angles.dtype)
        vol_samples, patches = generate_patches(volume,
                                                projection,
                                                angles,
                                                patch_size = self.config.model.patch_size, 
                                                n_points = self.config.training.num_points,
                                                patch_scale = self.patch_scale,
                                                boundary = self.config.training.boundary,
                                                discrete_sampling = self.config.training.discrete_sampling,
                                                nlims= zlim_values,)
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
            #print(patches.dtype)
            #print(angle_info.dtype)
            vol_est = self.model(patches,angle_info)
        else:
            vol_est = self.model(patches)

        # if output is a tuple (the second element is the auxiliary loss)
        if type(vol_est) is tuple:
            auxiliary_loss = vol_est[1]
            vol_est = vol_est[0][:,0]            
        else:
            vol_est = vol_est[:,0]
            auxiliary_loss = 0
        
        #print(auxiliary_loss)

        #print(vol_est.dtype)
        #print(vol_samples.dtype)

        loss = self.criterion(vol_est, vol_samples) + auxiliary_loss

        return loss
    def train_step(self,vol_loader,projection_simulator,vol_deformer, device = 'cpu'):
        """
        Train the model for one epoch
        """
        self.model.train()
        train_loss_epoch = []
        for i, train_volumes in enumerate(vol_loader):
            train_data = projection_simulator.simulate_batch(train_volumes, 
                                                             vol_deformer= vol_deformer,
                                                             downsample = self.config.training.downsample,
                                                             downsample_factors = self.config.training.downsample_factor,
                                                             scale_data= self.config.training.scale_data,
                                                             device= device)
            volume = train_data[0]['volume'].to(device)
            #print(volume)
            if self.config.training.get_projection_prefiltered: 
                projection = [train_data[0]['projections'].to(device),train_data[0]['projections_prefiltered'].to(device) ]
            else:
                projection = train_data[0]['projections'].to(device)
            
            angles = train_data[0]['angles'].to(device)

            

            projection_filt = projection.clone()
            if self.config.use_2D_filters:
                projection_filt = self.filter_2D(projection_filt)
            if self.config.filter_projections:
                if self.config.ramp.use_learnable_ramp is False:
                    with torch.no_grad():
                        projection_fiter = self.ramp(projection_filt.shape[2])
                        projection_filt = custom_ramp_fft(projection_filt,projection_fiter)
                else:
                    projection_fiter = self.ramp(projection_filt.shape[2])
                    projection_filt = custom_ramp_fft(projection_filt,projection_fiter)
            
            self.optimizer.zero_grad()
            loss = self.compute_loss(volume,projection_filt,angles)
            loss.backward()
            self.optimizer.step()
            train_loss_epoch.append(loss.item()) 
        self.scheduler.step()
        self.train_loss.append(np.mean(train_loss_epoch))


    def validate(self, valid_loader, device = 'cpu'):
        """
        valid loader contains the filtered projections
        """

        with torch.no_grad():
            valid_loss_epc = []
            self.model.eval()
            for j, data in enumerate(valid_loader):
                volume = data['volume'][0].to(device)
                if self.config.training.get_projection_prefiltered: 
                    projection = [data['projections'][0].to(device),data['projections_prefiltered'][0].to(device) ]
                else:
                    projection = data['projections'][0].to(device)
                angles = data['angles'][0].to(device)

                projection_filt = projection.clone()
                if self.config.use_2D_filters:
                        projection_filt = self.filter_2D(projection_filt)
                if self.config.filter_projections:
                    projection_fiter = self.ramp(projection_filt.shape[2])
                    projection_filt = custom_ramp_fft(projection_filt,projection_fiter) 


                loss = self.compute_loss(volume,projection_filt,angles)
                valid_loss_epc.append(loss.item())
            self.valid_loss.append(np.mean(valid_loss_epc))