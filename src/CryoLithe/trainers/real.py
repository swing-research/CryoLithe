import torch
import numpy as np
from .base import Trainer
from ..utils.utils import  custom_ramp_fft

class TrainerReal(Trainer):
    def train_step(self,data_loader, device = 'cpu', wandb_run = None):
        """
        Train the model for one epoch
        """
        self.model.train()
        train_loss_epoch = []
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

            if self.config.training.use_amp:
                #print('Using AMP')
                scaler = torch.cuda.amp.GradScaler()
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self.compute_loss(volume.half(),projection_filt.half(),angles.half(),zlims = zlim_values)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:            
                self.optimizer.zero_grad()
                loss = self.compute_loss(volume,projection_filt,angles,zlims = zlim_values)
                loss.backward()
                self.optimizer.step()
            train_loss_epoch.append(loss.item()) 
        self.scheduler.step()
        self.train_loss.append(np.mean(train_loss_epoch))   
        if wandb_run is not None:
            tr_loss = np.mean(train_loss_epoch)
            wandb_run.log({"train_loss": tr_loss}, step=len(self.train_loss)) 


    def validate(self, valid_loader, device = 'cpu', wandb_run = None):
        """
        valid loader contains the filtered projections
        """

        with torch.no_grad():
            valid_loss_epc = []
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

                #print('valid loss')
                #print(loss)
                valid_loss_epc.append(loss.item())
            self.valid_loss.append(np.mean(valid_loss_epc))
            if wandb_run is not None:
                val_loss = np.mean(valid_loss_epc)
                wandb_run.log({"valid_loss": val_loss}, step=len(self.valid_loss))