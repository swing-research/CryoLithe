# End-to-end localized deep learning for Cryo-ET 	
Official repo for CryoLithe ([paper](https://arxiv.org/abs/2501.15246))

CryoLithe is a supervised machine learning method to directly reconstruct the tomogram from aligned cryo-ET tilt series. The methods is trained on real measurements using FBP+cryo-CARE+IsoNet reconstructions as the reference. The network exploits the imaging geometry to extract small patches from the tilt series to recover the volume. Thus it is practically robust to various data distirbutions. The method provides FBP+cryo-CARE+IsoNet type of reconstructions in a fraction of the time.


## Updates 
- 20.04.2025: 
    - New models that can recover the volume from arbitrary number of tilt series. 
    - Update the pytorch version to 2.6.0 (the code was tested with 2.6.0)
    - Update the README file to include the new models and the new requirements.
    - Multi-GPU infernence is now supported.
- 05.08.2025
    - Added support to reconstruct a list of volumes from a single yaml file.
    - Added a new script `super-list.py` to run the model on a list of projections.

## Installation
Download the repository using the command:
```bash
git clone git@github.com:swing-research/CryoLithe.git
```


Create a new conda environment using the command:
```bash
conda create -n CryoLithe python=3.9
```



Activate the environment using the command:
```bash
conda activate CryoLithe
```
Install PyTorch 2.6 (or a compatible version). The code was tested with PyTorch 2.6
```bash
pip3 install torch torchvision torchaudio
```


Install the required packages using the command:

```bash
pip install -r requirements.txt
```

## Downloading the trained models
The models are stored in switchdrive and can be downloaded using the provided download.sh script:
```bash
bash download.sh
```

This will download the trained models and place them in the `trained_models` directory. It should contain the following files:
- `checkpoint.pth` - the trained model
- `config.json` - the configuration file used to train the model contains the model architecture and hyperparameters

Currently, we provide two models:
 - 'sliceset' - trained to recover the volume one voxel at a time
 - 'sliceset_wavelet' - trained to recover the wavelet coefficients of the volume

 **Note**: The wavelet model is 8x faster than the sliceset model. However, the reconstruction looks sligthly low resolution compared to the 
 sliceset model
## Running the model

We are actively working on extending the model to support arbitrary tilt series, which will be released in an upcoming update.

The script 'super.py' is used to run the trained model on any new projection of choice.  The script requires a configuration file that contains the necessary information to run the model.
The configuration file is a yaml file that contains the following fields:
 - 'model_dir' - path to the directory containing the trained model
 - 'proj_file' - path to the projection file
 - 'angle_file' - path to the angles file
 - 'save_dir' - path to the directory where the output will be saved
 - 'save_name' - name of the output volume
 - 'device' - device to run the model on (cpu or cuda)
 - 'downsample_projections' - Whether to downsample the projections or not
 - 'downsample_factor' - factor by which to downsample the volume
 - 'anti_alias' - whether to apply anti-aliasing to the projections or not
 - 'N3' - The size of the volume along the z-axis
 - 'batch_size' - batch size to use when running the model


The script can be run using the following command:
```bash
python3 super.py --config <path_to_config_file>
```

A sample yaml file is provided as 'ribo80.yaml' which contains the necessary information to run the model on the ribosome dataset.

## Running the model on the ribosome dataset

Download the ribosome dataset using the provided script:
```bash
bash download_ribosome.sh
```
This will download the ribosome dataset and place it in the `data` directory. The dataset contains the following files:
- `projections.mrcs` - the projections of the ribosome dataset
- `angles.tlt` - the angles of the projections

The data is downloaded from the EMPIAR 10045 dataset and is a subset of the full dataset.

To run the script, use the following command:
```bash
python3 super.py --config ribo80.yaml
```

## Using the Wavelet Model
Run the script using the following command:
```bash
python3 super.py --config ribo80_wavelet.yaml
```

## Running the model on a list of projections
The script `super-list.py` is used to run the trained model on a list of projections. Additionally, we provide a yaml file that can run the model on a list of projections.  In the `ribo80_list.yaml` file, you can specify multiple projection files, angle files, save names and N3 values for each projection. The script will then process each set of files in the list and save the corresponding volumes. Note that in the example yaml file, we are running the model on the same data twice, but you can modify it to have different projection data. You need to change the following fields in the yaml file:
- `proj_file` - list of paths to the projection files
- `angles_file` - list of paths to the angles files
- `save_name` - list of names for the output volumes
- `N3` - list of sizes for the volumes along the z-axis

You can run the script using the following command:
```bash
python3 super-list.py --config ribo80_list.yaml
```