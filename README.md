# Supervised Approach to CryoET Reconstructions

## Installation
Download the repository using the command:
```bash
git clone git@github.com:swing-research/supervised-cryoET-inference.git
```


Create a new conda environment using the command:
```bash
conda create -n super python=3.8
```



Activate the environment using the command:
```bash
conda activate super
```
Install a version of PyTorch that is compatible with your system using the command:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
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

## Running the model
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
