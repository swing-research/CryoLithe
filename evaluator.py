"""
Load the model and run on the given data
"""
from ml_collections import config_dict
import json
import torch
from models import get_model, model_wrapper
from utils.utils import custom_ramp_fft, generate_patches_from_volume_location
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist

class Evaluator:
    def __init__(self, model_path , device):

        configs = config_dict.ConfigDict(json.load(open(model_path + '/config.json')))
        self.configs = configs
        self.device = device

        self.n_projections = configs.data.n_projections

        checkpoint = torch.load(model_path + '/checkpoint.pth',map_location=torch.device(device))

        
        model = get_model(n_projections = self.n_projections, **configs.model).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])


        if configs.filter_projections:
            ramp = checkpoint['ramp']
            if ramp is not None:
                ramp = ramp.to(device)
        else:
            ramp = None

        if configs.training.learn_patch_scale:
            patch_scale = checkpoint['patch_scale'].to(device)
            patch_scale = patch_scale.detach()
        else:
            patch_scale = torch.tensor([configs.training.patch_scale_init]).to(device)


        if configs.use_2D_fitlers or configs.use_2D_filters:
            filter_2D =checkpoint['filter_2D'].to(device)
        else: 
            filter_2D = None


        self.model = model
        self.ramp = ramp
        self.patch_scale = patch_scale
        self.filter_2D = filter_2D

    def  full_reconstruction_slice(self, projection, angles, N3_scale = 0.5,batch_size = int(4e4) ,N3 = None,):
        """
        projections: list of projections or projections of shape (n_projections, N_1, N_2)
        angles: list of angles or angles of shape (n_projections) in degrees
        """

        projection_filt = self.filter_projections(projection)
        angles_t = torch.tensor(angles, dtype=torch.float32, device=self.device)*torch.pi/180


        N1,N2 = projection_filt.shape[-2], projection_filt.shape[-1]
        if N3 is None:
            N3 = int(max(N1,N2)*N3_scale)

        
        z_index_set = torch.arange(projection_filt.shape[-1]//2-N3//2, projection_filt.shape[-1]//2+N3//2)

        vol_dummy = torch.zeros((100,100,50),dtype=torch.float32,device=self.device)

        vol_full_est = np.zeros((N1,N2,N3),dtype=np.float32)

        x_index = torch.linspace(-1,1,N1)
        y_index = torch.linspace(-1,1,N2)
        z_index_values = torch.linspace(-1,1,N2)

        scale = np.ones(3)
        scale[2] = vol_dummy.shape[2]/vol_dummy.shape[1]    
        xx_test, yy_test = torch.meshgrid(x_index,y_index ,indexing= 'ij')  
        points_2D = torch.cat((yy_test.unsqueeze(-1),xx_test.unsqueeze(-1)),dim=2).to(self.device)
        points_3D = torch.zeros((points_2D.shape[0],points_2D.shape[1],3)).to(self.device)
        points_3D[:,:,1:] = points_2D 
        with torch.no_grad():
            self.model.eval()
            for rel_index,z_index in enumerate(tqdm(z_index_set, desc='z_index')): 

                z_index = z_index_values[z_index]
                points_3D[:,:,0] = z_index

                point3D_vec = points_3D.view(-1,3)

                point_loader = DataLoader(point3D_vec,shuffle=False,batch_size=batch_size, num_workers=8)
               
                vol_est_set = []
                for points in point_loader:
                    vol_true, projection_patches = generate_patches_from_volume_location(points, vol_dummy ,
                                                                                            projection_filt,
                                                                                            angles_t,
                                                                                            patch_size = self.configs.model.patch_size,
                                                                                        scale=scale,
                                                                                        patch_scale= self.patch_scale) 

                    vol_est = self.model(projection_patches)[:,0].detach().cpu().numpy() 
                    vol_est_set.append(vol_est)
                
                vol_est =np.concatenate(vol_est_set).reshape(N1,N2)
                vol_full_est[:,:,rel_index] = vol_est
                
        return vol_full_est
    

    def full_reconstruction(self, projection, angles, N3_scale = 0.5,batch_size = int(4e4) ,N3 = None, num_workers =4):
        """
        projections: list of projections or projections of shape (n_projections, N_1, N_2)
        angles: list of angles or angles of shape (n_projections) in degrees
        """

        projection_filt = self.filter_projections(projection)
        angles_t = torch.tensor(angles, dtype=torch.float32, device=self.device)*torch.pi/180

        N1,N2 = projection_filt.shape[-2], projection_filt.shape[-1]
        if N3 is None:
            N3 = int(max(N1,N2)*N3_scale)

        z_index_set = torch.arange(projection_filt.shape[-1]//2-N3//2, projection_filt.shape[-1]//2+N3//2)

        vol_dummy = torch.zeros((100,100,50),dtype=torch.float32,device=self.device)

        vol_full_est = np.zeros((N1,N2,N3),dtype=np.float32)

        x_index = torch.linspace(-1,1,N1)
        y_index = torch.linspace(-1,1,N2)
        z_index_values = torch.linspace(-1,1,N2)
        z_index = z_index_values[projection_filt.shape[-1]//2 - N3//2:projection_filt.shape[-1]//2 + N3//2]

        scale = np.ones(3)
        scale[2] = vol_dummy.shape[2]/vol_dummy.shape[1]    
        zz_test,xx_test, yy_test = torch.meshgrid(z_index,x_index,y_index ,indexing= 'ij')  
        points_3D = torch.cat((zz_test.unsqueeze(-1),yy_test.unsqueeze(-1),xx_test.unsqueeze(-1)),dim=3)
        points_3D = points_3D.reshape(-1,3)
        point_loader = DataLoader(points_3D,shuffle=False,batch_size=batch_size,num_workers=num_workers)
        with torch.no_grad():
            self.model.eval()
            vol_est_set = []

            for points in tqdm(point_loader):
                points = points.to(self.device)

                vol_true, projection_patches = generate_patches_from_volume_location(points, vol_dummy ,
                                                                                    projection_filt,
                                                                                    angles_t,
                                                                                    patch_size = self.configs.model.patch_size,
                                                                                scale=scale,
                                                                                patch_scale= self.patch_scale) 

                vol_est = self.model(projection_patches)[:,0].detach().cpu().numpy() 
                vol_est_set.append(vol_est)
            vol_est =np.concatenate(vol_est_set).reshape(N3,N1,N2)
            vol_full_est = np.moveaxis(vol_est,0,-1)
                
        return vol_full_est
    
    def full_reconstruction_distribute(self, projection, angles, 
                                       N3_scale = 0.5,
                                       batch_size = int(4e4) ,
                                       N3 = None, 
                                       num_workers =4,
                                       gpu_ids = [0,1]):
        """
        projections: list of projections or projections of shape (n_projections, N_1, N_2)
        angles: list of angles or angles of shape (n_projections) in degrees
        """

        projection_filt = self.filter_projections(projection)
        angles_t = torch.tensor(angles, dtype=torch.float32, device=self.device)*torch.pi/180

        N1,N2 = projection_filt.shape[-2], projection_filt.shape[-1]
        if N3 is None:
            N3 = int(max(N1,N2)*N3_scale)

        z_index_set = torch.arange(projection_filt.shape[-1]//2-N3//2, projection_filt.shape[-1]//2+N3//2)

        vol_dummy = torch.zeros((100,100,50),dtype=torch.float32,device=self.device)

        vol_full_est = np.zeros((N1,N2,N3),dtype=np.float32)

        x_index = torch.linspace(-1,1,N1)
        y_index = torch.linspace(-1,1,N2)
        z_index_values = torch.linspace(-1,1,N2)
        z_index = z_index_values[projection_filt.shape[-1]//2 - N3//2:projection_filt.shape[-1]//2 + N3//2]

        scale = np.ones(3)
        scale[2] = vol_dummy.shape[2]/vol_dummy.shape[1]    
        zz_test,xx_test, yy_test = torch.meshgrid(z_index,x_index,y_index ,indexing= 'ij')  
        points_3D = torch.cat((zz_test.unsqueeze(-1),yy_test.unsqueeze(-1),xx_test.unsqueeze(-1)),dim=3)
        points_3D = points_3D.reshape(-1,3)


        modl_wrapper = model_wrapper(self.model,
                                     projections = projection_filt,
                                     angles = angles_t,
                                     volume_dummy= vol_dummy,
                                     patch_scale = self.patch_scale,
                                     scale = scale,
                                     configs = self.configs)

        modl_wrapper = modl_wrapper.to(self.device)
        modl_wrapper = torch.nn.DataParallel(modl_wrapper, device_ids = gpu_ids)

        point_loader = DataLoader(points_3D,shuffle=False,batch_size=batch_size,num_workers=num_workers)

        with torch.no_grad():
            modl_wrapper.eval()
            vol_est_set = []
            for points in tqdm(point_loader):

                vol_est = modl_wrapper(points)[:,0].detach().cpu().numpy()
                vol_est_set.append(vol_est)
            vol_est =np.concatenate(vol_est_set).reshape(N3,N1,N2)
            vol_full_est = np.moveaxis(vol_est,0,-1)
        return vol_full_est




    
    def orthogonal_reconstruction(self, projection, angles, N3_scale = 0.5,batch_size = int(4e4) ,N3 = None,):
        """
        Reconstruction of the orthogonal slices
        """

        projection_filt = self.filter_projections(projection)
        angles_t = torch.tensor(angles, dtype=torch.float32, device=self.device)*torch.pi/180


        N1,N2 = projection_filt.shape[-2], projection_filt.shape[-1]
        if N3 is None:
            N3 = int(max(N1,N2)*N3_scale)

        
        z_index_set = torch.arange(projection_filt.shape[-1]//2-N3//2, projection_filt.shape[-1]//2+N3//2)

        vol_dummy = torch.zeros((100,100,50),dtype=torch.float32,device=self.device)

        vol_ortho_ests = []

        x_index = torch.linspace(-1,1,N1)
        y_index = torch.linspace(-1,1,N2)
        z_index_values = torch.linspace(-1,1,N2)

        scale = np.ones(3)
        scale[2] = vol_dummy.shape[2]/vol_dummy.shape[1]    
        xx_test, yy_test = torch.meshgrid(x_index,y_index ,indexing= 'ij')  
        points_2D = torch.cat((yy_test.unsqueeze(-1),xx_test.unsqueeze(-1)),dim=2).to(self.device)
        points_3D = torch.zeros((points_2D.shape[0],points_2D.shape[1],3)).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            points_3D[:,:,1:] = points_2D 
            points_3D[:,:,0] = 0
            point3D_vec = points_3D.view(-1,3)
            point_loader = DataLoader(point3D_vec,shuffle=False,batch_size=batch_size)

            x_y_slice = self.slice_eval(point_loader, projection_filt, angles_t, scale, vol_dummy)
            x_y_slice = np.concatenate(x_y_slice).reshape(N1,N2)

            points_3D[:,:,:2] = points_2D 
            points_3D[:,:,2] = 0
            point3D_vec = points_3D.view(-1,3)
            point_loader = DataLoader(point3D_vec,shuffle=False,batch_size=batch_size)
            x_z_slice = self.slice_eval(point_loader, projection_filt, angles_t, scale, vol_dummy)
            x_z_slice = np.concatenate(x_z_slice).reshape(N1,N2)

            points_3D[:,:,0] = points_2D[:,:,0]
            points_3D[:,:,2] = points_2D[:,:,1]
            points_3D[:,:,1] = 0

            point3D_vec = points_3D.view(-1,3)
            point_loader = DataLoader(point3D_vec,shuffle=False,batch_size=batch_size)

            y_z_slice = self.slice_eval(point_loader, projection_filt, angles_t, scale, vol_dummy)
            y_z_slice = np.concatenate(y_z_slice).reshape(N1,N2)


            output_dict = {'x_y_slice':x_y_slice, 'x_z_slice':x_z_slice, 'y_z_slice':y_z_slice}
            return output_dict
         
            


    def slice_eval(self, pointloader, projection,angles, scale,vol_dummy):
        """
        Evaluate the slices
        """

        vol_est_set = []
        for points in tqdm(pointloader):



            vol_true, projection_patches = generate_patches_from_volume_location(points, vol_dummy ,
                                                                                    projection,
                                                                                    angles,
                                                                                    patch_size = self.configs.model.patch_size,
                                                                                scale=scale,
                                                                                patch_scale= self.patch_scale) 

            vol_est = self.model(projection_patches)[:,0].detach().cpu().numpy() 
            vol_est_set.append(vol_est)
        return vol_est_set




    def filter_projections(self,projections):
        """
        Filter the projections
        """

        if type(projections) == list:
            proj_filt = []

            for proj in projections:
                proj_filt.append(self.filter_single_projection(proj))
        else:
            proj_filt = self.filter_single_projection(projections)

        return proj_filt
            

    def filter_single_projection(self,projection):
        """
        Filter a single projection
        """
        proj_t = torch.tensor(projection, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.configs.use_2D_filters:
                proj_t = self.filter_2D(proj_t)
            
            if self.configs.filter_projections:
                ramp_filt = self.ramp(proj_t.shape[-1])
                proj_t = custom_ramp_fft(proj_t, ramp_filt, use_splits= True)

            return proj_t
    



        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import mrcfile
    import torch
    DEVICE = 1
    eval = Evaluator(model_path = './trained_models/real_trained/' , device = DEVICE)

    path = '/home/kishor0000/Work/cryoET/ET_data_supervised/10045-80S/IS002_291013_005.mrcs'
    proj = mrcfile.open(path).data
    proj = proj - np.mean(proj)
    proj = proj/np.std(proj)


    angles = np.loadtxt('/home/kishor0000/Work/cryoET/ET_data_supervised/10045-80S/angle_5.rawtlt')

    DOWNSAMPLE = True 
    DOWNSAMPLE_FACTOR = 0.25
    proj_ds_set = []
    if DOWNSAMPLE:
        for proj in proj:
            proj_t = torch.tensor(proj,device = DEVICE,dtype =torch.float32)
            proj_ds = torch.nn.functional.interpolate(proj_t[None,None] , 
                                                    scale_factor=DOWNSAMPLE_FACTOR, 
                                                    align_corners=False,
                                                    antialias = True,
                                                    mode='bicubic').squeeze()
            proj_ds_set.append(proj_ds.cpu().numpy())
            
        proj_real = np.array(proj_ds_set)


    op = eval.orthogonal_reconstruction(proj_real,angles,512)
