"""
Dataset class for loading the real volumes along wth the coresponding projections and their angles
"""

import numpy as np
import torch
import mrcfile
from torch.utils.data import Dataset
import time


class RealVolumes(Dataset):
    """
    Loads the volumes from the simulator dataset
    """
    def __init__(self, root_dir: str ,
                vol_paths: list,
                projection_paths: list,
                angle_paths: list,
                normalize_type: str = 'vol',
                z_lims_list: list = None,
                randomize_projections: bool = False,
                full_randomize: bool = False,
                projection_paths_odd: list = None,
                projection_paths_even: list = None,
                remove_projections_wedge: bool = False,
                max_remove_projections_wedge: int = 0,
                remove_projections_random: bool = False,
                max_remove_projections_random: int = 0,

                cache: bool = False):
        """
        Args:
            root_dir (string): Directory with all the folders.
            vol_paths (list): List of paths to the volumes
            projection_paths (list): List of paths to the projections
            angle_paths (list): List of paths to the angles
            normalize_type (string): Type of normalization to be applied to the volumes
            cache (bool): If True, the volumes are cached in memory
            full_randomize (bool): If True, then each projection is randomly 
            selected from the three projections provided

        """

        self.root_dir = root_dir
        self.normalize_type = normalize_type
        self.vol_paths = vol_paths
        self.projection_paths = projection_paths
        self.angle_paths = angle_paths
        self.cache = cache
        self.z_lims_list = z_lims_list
        self.projection_paths_odd = projection_paths_odd
        self.projection_paths_even = projection_paths_even
        self.randomize_projections = randomize_projections
        self.full_randomize = full_randomize
        self.remove_projections_wedge = remove_projections_wedge
        self.max_remove_projections_wedge = max_remove_projections_wedge
        self.remove_projections_random = remove_projections_random
        self.max_remove_projections_random = max_remove_projections_random

        if self.cache:
            print("Preloading volumes")
            self.data_list= []
            for i, (vol_path, proj_path, angle_path) in enumerate(zip(self.vol_paths, self.projection_paths, self.angle_paths)):
                print(f"Loading {vol_path}")

                vol = mrcfile.open(self.root_dir + vol_path, mode='r').data
                vol = vol.astype(np.float32)
                vol = self.data_normalize(vol)
                vol = np.moveaxis(vol, 0, 2).copy()

                #print(f"Loading {proj_path}")
                proj = mrcfile.open(self.root_dir + proj_path, mode='r').data
                proj = proj.astype(np.float32)
                proj = self.data_normalize(proj)

                #print(f"Loading {angle_path}")
                angles = np.loadtxt(self.root_dir + angle_path)*np.pi/180
                angles = angles.astype(np.float32)


                # load the projection paths for odd and even if they are provided

                if projection_paths_odd is not None:
                    if projection_paths_odd[i] is None:
                        proj_odd = None
                    else:
                        proj_odd = mrcfile.open(self.root_dir + projection_paths_odd[i], mode='r').data
                        proj_odd = proj_odd.astype(np.float32)
                        proj_odd = self.data_normalize(proj_odd)

                if projection_paths_even is not None:
                    if projection_paths_even[i] is None:
                        proj_even = None
                    else:
                        proj_even = mrcfile.open(self.root_dir + projection_paths_even[i], mode='r').data
                        proj_even = proj_even.astype(np.float32)
                        proj_even = self.data_normalize(proj_even)

                

                data = {'volume': vol,
                         'projection_full' : proj,
                           'angles': angles, 'vol_name': vol_path}
                
                if projection_paths_odd is not None:
                    data['projection_odd'] = proj_odd
                if projection_paths_even is not None:
                    data['projection_even'] = proj_even
                if self.z_lims_list is not None:
                    data['z_lims'] = self.z_lims_list[i]
                self.data_list.append(data)

    def data_normalize(self, data):
        """
        Normalize the data
        TODO: Add more normalization options
        """
        data = (data - np.mean(data))/np.std(data)
        return data

    def __len__(self):
        return len(self.vol_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # load ref volume:

        if self.cache:
            #print("Loading from cache")
            start = time.time()
            data = self.data_list[idx].copy()

            # if self.projection_paths_odd is not None or self.projection_paths_even is not None 
            # the load one of the three projections randomly

            if self.randomize_projections:
                # If odd or even projections are None, then load the full projection
                if data['projection_odd'] is None or data['projection_even'] is None:
                    data['projection'] = data['projection_full']

                    # remove the odd and even projections
                    data.pop('projection_odd')
                    data.pop('projection_even')

                else:
                    if self.full_randomize is False:
                        rand = np.random.randint(0,3)
                        if rand == 0:
                            data['projection'] = data['projection_full']
                        elif rand == 1:
                            data['projection'] = data['projection_odd']
                        else:
                            data['projection'] = data['projection_even']
                    else:
                        porj_len = len(data['projection_full'])
                        rand_full  = np.random.randint(0,4)
                        if rand_full == 0:
                            data['projection'] = data['projection_full']
                        elif rand_full == 1:
                            data['projection'] = data['projection_odd']
                        elif rand_full == 2:
                            data['projection'] = data['projection_even']
                        else:
                            rand = np.random.randint(0,2,porj_len)
                            proj_mix = np.zeros_like(data['projection_full'])
                            proj_mix[rand==0] = data['projection_odd'][rand==0]
                            proj_mix[rand==1] = data['projection_even'][rand==1]
                            data['projection'] = proj_mix
            else:
                data['projection'] = data['projection_full']

            end = time.time()
            #print(f"Time taken to load from cache: {end-start}")
        else:
            start = time.time()
            vol = mrcfile.open(self.root_dir + self.vol_paths[idx], mode='r').data
            vol = vol.astype(np.float32)
            projection  = mrcfile.open(self.root_dir + self.projection_paths[idx], mode='r').data
            projection = projection.astype(np.float32)
            angles = np.loadtxt(self.root_dir + self.angle_paths[idx])*np.pi/180
            angles = angles.astype(np.float32)
            vol = np.moveaxis(vol, 0, 2)
            #Normalize the data
            vol = (vol - np.mean(vol))/np.std(vol)
            projection = (projection - np.mean(projection))/np.std(projection)
            end = time.time()
            #print(f"Time taken to load from disk: {end-start}")
            start = time.time()
            data = {'volume': vol, 'projection': projection, 'angles': angles, 'vol_name': self.vol_paths[idx]}
            if self.z_lims_list is not None:
                data['z_lims'] = self.z_lims_list[idx]
            end = time.time()
            #print(f"Time taken create the dict: {end-start}")


        data = self.remove_projections(data)        
        return data
    
    def remove_projections(self, data):
        """"
        Randomly remove some projections from the data
        this can be done at random or remove certain number of projections in a range.
        """


        
        if self.remove_projections_wedge:
            remove_wedge = np.random.randint(0,2)

            if remove_wedge:
                n_projections = len(data['projection'])
                n_remove = np.random.randint(1,self.max_remove_projections_wedge)
                # remove first or last n_remove projections randomly
                rand = np.random.randint(0,2)
                #print(n_remove)
                if rand:
                    data['projection'] = data['projection'][n_remove:]
                    data['angles'] = data['angles'][n_remove:]
                else:
                    data['projection'] = data['projection'][:-n_remove]
                    data['angles'] = data['angles'][:-n_remove]
        if self.remove_projections_random:
            drop_random = np.random.randint(0,2)
            if drop_random:
                n_projections = len(data['projection'])
                n_remove = np.random.randint(0,self.max_remove_projections_random)
                rand = np.random.randint(0,n_projections,n_remove)
                data['projection'] = np.delete(data['projection'],rand,0)
                data['angles'] = np.delete(data['angles'],rand,0)
        return data



