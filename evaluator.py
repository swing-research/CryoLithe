"""
Load the model and run on the given data
"""
from ml_collections import config_dict
import json
import torch
from models import get_model, model_wrapper
from utils.utils import custom_ramp_fft
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils.wavelet_utils import wavelet_multilevel_decomposition, wavelet_multilevel_reconstruction



class Evaluator:
    def __init__(self, model_path , device):

        configs = config_dict.ConfigDict(json.load(open(model_path + '/config.json')))
        self.configs = configs
        self.device = device

        self.n_projections = configs.data.n_projections

        checkpoint = torch.load(model_path + '/checkpoint.pth',map_location=torch.device(device), weights_only=False)

        
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


    def generate_points(self,n1,n2,n3):
        """
        Generate points in 3D space
        """

        n = max(n1,n2,n3)
        scale = np.ones(3)
        scale[0] = n1/n
        scale[1] = n2/n
        scale[2] = n3/n

        x_index = torch.linspace(-1,1,n1)
        x_index  = x_index*scale[0]
        y_index = torch.linspace(-1,1,n2)
        y_index  = y_index*scale[1]
        z_index = torch.linspace(-1,1,n3)
        z_index  = z_index*scale[2]

        zz,yy,xx = torch.meshgrid(z_index,y_index,x_index)
        points = torch.stack([zz,yy,xx],dim=3).reshape(-1,3)

        return points,scale
    

    def pre_process(self, projection, angles,  N3_scale = 0.5, N3 =None):
        """
        projections: projections of shape (n_projections, N_1, N_2)
        angles: angles of shape (n_projections) in degrees
        """


        projection_filt = self.filter_projections(projection)
        angles_t = torch.tensor(angles, dtype=torch.float32, device=self.device)*torch.pi/180

        N1,N2 = projection_filt.shape[-2], projection_filt.shape[-1]
        if N3 is None:
            N3 = int(max(N1,N2)*N3_scale)

        vol_dummy = torch.randn(N1,N2,N3,dtype=torch.float32,device=self.device)


        if self.configs.training.use_wavelet_trainer:
            vol_wavelet = wavelet_multilevel_decomposition(vol_dummy, 
                                                        self.configs.training.wavelet, 
                                                        levels = self.configs.training.wavelet_levels)
            vol_lp = vol_wavelet[0]
            N1,N2,N3 = vol_lp.shape

        
        points,scale = self.generate_points(N1,N2,N3)

        return projection_filt, angles_t, points, vol_dummy, N1,N2,N3,scale


    def reconstruct(self, projection, 
                    angles, 
                    N3_scale = 0.5,
                    batch_size = int(4e4) ,
                    N3 = None, 
                    num_workers =4,
                    gpu_ids = None):
        """
        projections: projections of shape (n_projections, N_1, N_2)
        angles: angles of shape (n_projections) in degrees
        """


        projection_filt, angles_t, points, vol_dummy, N1,N2,N3,scale = self.pre_process(projection, 
                                                                                  angles, 
                                                                                  N3_scale = N3_scale, 
                                                                                  N3 = N3)
        
        modl_wrapper = model_wrapper(self.model,
                                     projections = projection_filt,
                                     angles = angles_t,
                                     volume_dummy= vol_dummy,
                                     patch_scale = self.patch_scale,
                                     scale = scale,
                                     configs = self.configs)

        modl_wrapper = modl_wrapper.to(self.device).half()
        if gpu_ids is not None:
            modl_wrapper = torch.nn.DataParallel(modl_wrapper, device_ids = gpu_ids)
        point_loader = DataLoader(points.half(),shuffle=False,batch_size=batch_size,num_workers=num_workers)


        with torch.no_grad():
            modl_wrapper.eval()
            vol_est_set = []
            for points in tqdm(point_loader):
                if gpu_ids is None:
                    points = points.to(self.device)
                vol_est = modl_wrapper(points).permute(1,0).cpu().numpy()
                vol_est_set.append(vol_est)
            v_est_set_np = np.moveaxis(np.moveaxis(np.concatenate(vol_est_set,axis=1).reshape(-1,N3,N1,N2),1,-1),2,1)

        if self.configs.training.use_wavelet_trainer:   
            v_est_set_t = torch.tensor(v_est_set_np, dtype=torch.float32, device=self.device)
            vol_est_rec = wavelet_multilevel_reconstruction(v_est_set_t, 
                                                            wavelet= self.configs.training.wavelet).cpu().numpy()
        else:
            vol_est_rec = v_est_set_np[0]

        return vol_est_rec


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
