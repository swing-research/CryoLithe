"""
Script to load the traned model and reconstruct the volumes present in the yaml file
"""




import os
import yaml
import argparse
from evaluator import Evaluator
import mrcfile
import torch
import numpy as np

args = argparse.ArgumentParser(description="Load the trained model and reconstruct the volumes present in the yaml file")

args.add_argument("--config", type=str, help="Path to the yaml file")




if __name__ == "__main__":
    # Load the yaml file

    args = args.parse_args()
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    device = config["device"]
    model_path = config["model_dir"]
    batch_size = config["batch_size"]
    downsample = config["downsample_projections"]
    N3 = config["N3"]
    save_dir = config["save_dir"]
    save_name = config["save_name"]
    angles = np.loadtxt(config["angle_file"])

    # check if the parameter is preset
    if "num_workers" in config:
        num_workers = config["num_workers"]
        print('num_workers:',num_workers)
    else:
        num_workers = 0


    eval = Evaluator(model_path = model_path , device = device)

    proj_path = config["proj_file"]
    projection = mrcfile.open(proj_path, permissive=True).data
    projection = projection - np.mean(projection)
    projection = projection/np.std(projection)


    if len(angles) > 41:
        print("We resample the tilt-series and we need exactly 41 tilt at the moment.")
        ind_sangles = np.linspace(0,len(angles)-1,41).astype(int)
        angles = angles[ind_sangles]
        projection = projection[ind_sangles]

    if len(angles)<41:
        print("Impossible to work with less than 41 projections at the moment.")
        import sys
        sys.exit(0)

       
    
    if downsample:
        downsample_factor = config["downsample_factor"]
        anti_alias = config["anti_alias"]
        proj_ds_set = []
        for proj in projection:
            proj_t = torch.tensor(proj,device = device,dtype =torch.float32)
            proj_ds = torch.nn.functional.interpolate(proj_t[None,None] , 
                                                    scale_factor=downsample_factor, 
                                                    align_corners=True,
                                                    antialias = anti_alias,
                                                    mode='bicubic').squeeze()
            proj_ds_set.append(proj_ds.cpu().numpy())
            
        projection = np.array(proj_ds_set)



    # Zero pad the projections to make them square
    N1 = projection.shape[1]
    N2 = projection.shape[2]

    if N1>N2:
        pad = (N1-N2)//2
        projection = np.pad(projection,((0,0),(0,0),(pad,pad)))
    elif N2>N1:
        pad = (N2-N1)//2
        projection = np.pad(projection,((0,0),(pad,pad),(0,0)))

    if N3 > int(max(N1,N2)):
        print("Changed value of N3 to be same as max(N1,N2)")
        N3 =  int(max(N1,N2))

    





    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ortho_slice = eval.orthogonal_reconstruction(projection = projection,
                                                angles = angles,
                                                N3 = N3,
                                                N3_scale = 0.5,
                                                batch_size = batch_size)
    

    # output_dict = {'x_y_slice':x_y_slice, 'x_z_slice':x_z_slice, 'y_z_slice':y_z_slice}


    #Save the orthogonal slices as mrc files
    for key in ortho_slice.keys():
        save_path = os.path.join(save_dir,save_name + f"_{key}.mrc")
        out = mrcfile.new(save_path,overwrite = True)
        out.set_data(ortho_slice[key])
        out.close()

    vol = eval.full_reconstruction(projection = projection, 
                                   angles= angles,
                                   N3 = N3, 
                                   N3_scale = 0.5,
                                   batch_size = batch_size, 
                                   num_workers=num_workers)
    
    vol = np.moveaxis(vol,2,0)
    if N1 > N2:
        vol = vol[:,:,pad:-pad]
    elif N2 > N1:
        vol = vol[:,pad:-pad]
    
    save_path = os.path.join(save_dir,save_name)

    out = mrcfile.new(save_path,overwrite = True)
    out.set_data(vol)
    out.close()
