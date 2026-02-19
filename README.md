# End-to-end localized deep learning for Cryo-ET 	
Official repo for CryoLithe ([paper](https://arxiv.org/abs/2501.15246))

CryoLithe is a supervised machine learning method to directly reconstruct the tomogram from aligned cryo-ET tilt series. The methods is trained on real measurements using FBP+cryo-CARE+IsoNet and FBP+Icecream reconstructions as the reference. The network exploits the imaging geometry to extract small patches from the tilt series to recover the volume. Thus it is practically robust to various data distirbutions. The method provides FBP+cryo-CARE+IsoNet type of reconstructions in a fraction of the time.


## Updates 
- 18.04.2025: 
    - New models that can recover the volume from arbitrary number of tilt series. 
    - Update the pytorch version to 2.6.0 (the code was tested with 2.6.0)
    - Update the README file to include the new models and the new requirements.
    - Multi-GPU infernence is now supported.
- 05.08.2025
    - Added support to reconstruct a list of volumes from a single yaml file.
    - Added a new script `super-list.py` to run the model on a list of projections.
- 05.09.2025
    - Added new trained models that were trained on a larger dataset. 

## Installation
You can install CryoLithe using either [**conda**][#using-conda-recommended-if-you-already-use-conda] (recommended if you already use conda) or [**uv**][#using-uv] (lightning fast installer).

### Using conda (recommended if you already use conda):

Create a new conda environment using the command:
```bash
conda create -n CryoLithe python=3.9
```
Activate the environment using the command:
```bash
conda activate CryoLithe
```
Install PyTorch 2.6 (or a compatible version). The code was tested with PyTorch 2.6 and 2.8
```bash
pip3 install torch torchvision torchaudio
```

###  Install CryoLithe and its dependencies:
You can install CryoLithe directly from the GitHub repository using pip or clone the repository and install it locally.


####  Direct installation (recommended):
```bash
pip install git+https://github.com:swing-research/CryoLithe.git
```

To test the installation, run:
```bash
cryolithe --help
```
It should display the main commands: `reconstruct`, `download`, and `download-sample-data`


#### Local installation from cloned repository (for ones who want to modify the code):

Clone the repository using **one** of the following methods:

**HTTPS**
```bash
git clone https://github.com:swing-research/CryoLithe.git
```
**SSH (for users with SSH keys configured):**
```bash
git clone git@github.com:swing-research/CryoLithe.git
```


Install in editable mode:

```bash
pip install -e .
```
To test the installation, run:
```bash
cryolithe --help
```
It should display the two main commands: `reconstruct` and `download`



### Using uv: 
If you don't have uv installed, you can use the following command (if using a linux or macOS system), alternatively  you can follow the instructions in the  official [uv documentation](https://docs.astral.sh/uv/#installation):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
This will install uv in your system and you need to restart your terminal to take effect.

Once you have uv installed, there are multiple ways to install CryoLithe using uv. Here we will install it as a tool (uv terminology) which allows us to run the `cryolithe` command directly in the terminal without activating a virtual environment. But you uv creates a virtuall environtment internally to manage the dependencies of the tool. To install CryoLithe as a tool, run the following command:
```bash
uv tool install git+https://github.com/swing-research/CryoLithe.git
```

To test the installation, run:
```bash
cryolithe --help
```
It should display the main commands: `reconstruct`, `download`, and `download-sample-data`

To uninstall the tool, run:
```bash
uv tool uninstall cryolithe
```

## Downloading the trained models
The trained models are stored in Hugging Face and can be downloaded using the `cryolithe download` command. You can specify the local directory where you want to save the models using the `--local-dir` flag. For example, to save the models in a directory called `models`, run:

```bash
cryolithe download --local-dir /path/to/trained/model/
```
If you want it to be saved in the Hugging Face cache, then run:
```bash
cryolithe download
```

Note that this will create a file called .cryolithe.yaml in your home directory with the path to the models files. This is used by the reconstruct command to load models when you don't specify the path to the model directory. You can modify this file to change the path to the models if you want to move them to a different location. 
By default, each new `cryolithe download` updates `model_dir` in this file to the newly downloaded model path. If you want to keep the existing path, use:
```bash
cryolithe download --no-override-model-dir
```


Currently, we provide two models:
 - 'cryolithe_pixel' - trained to recover the volume one pixel at a time
 - 'cryolithe' - trained to recover the wavelet coefficients of the volume.


 **Note**: The wavelet model is 8x faster than the sliceset model. However, the reconstruction looks sligthly low resolution compared to the 
 sliceset model

## Downloading sample data
Download sample tilt-series data from Hugging Face dataset repo:
```bash
cryolithe download-sample-data
```
By default this downloads to `./cryolithe-sample-data` in the directory where you run the command. You can also provide your own path:
```bash
cryolithe download-sample-data --local-dir /path/to/data
```


# Reconstructing the tomograms using the trained models
Once the models are downloaded, you can use them to reconstruct the tomograms from the projections. This is done using the  `reconstruct` command of the `cryolithe` package. The command can take a yaml file as input or you can specify the arguments directly in the command line.  An example yaml file is provided in the repository as [`ribo80.yaml`](ribo80.yaml) which contains the necessary information to run the model on the ribosome dataset.
**Note**: Make sure to update the paths in the yaml file to point to the correct locations of the projection and angle files and model directory on your system before running the command.

```bash
cryolithe reconstruct --config ribo80.yaml
```
You can create your own yaml file with the necessary information to run the model on your data. 


Alternatively, a convenient way would be to use the command line arguments to run the model. For example, to run the model on a sample dataset, you can use the following command:
```bash
cryolithe reconstruct \ 
    --model-dir /path/to/trained/model/ 
    --proj-file /path/to/projections.mrc \
    --angle-file /path/to/angles.tlt \
    --save-dir /path/to/save/directory/ \
    --save-name output_volume.mrc \
    --device 0 \
    --n3 256 \ # The size of the volume along the z-axis, you can modify this based on your data
    --batch-size 100000  # Depends on the memory of your GPU
```

You can update other arguments as well based on your data and preferences. For all the parameters and their descriptions, you can run the following command:
```bash
cryolithe reconstruct --help
```

You don't need to specify the path to the model directory if you have downloaded the models using the `cryolithe download` command, as the path to the models is stored in the .cryolithe.yaml file in your home directory. In that case, you can simply run:
```bash
cryolithe reconstruct \ 
    --proj-file /path/to/projections.mrc \
    --angle-file /path/to/angles.tlt \
    --save-dir /path/to/save/directory/ \
    --save-name output_volume.mrc \
    --device 0 \
    --n3 256 \ # The size of the volume along the z-axis, you can modify this based on your data
    --batch-size 100000  # Depends on the memory of your GPU
``

If the model directory is not provided, the reconstruct code will choose the wavelet model by default. If you want to use the pixel model, you can use the flag `--pixel` to specify that you want to use the pixel model. For example:
```bash
cryolithe reconstruct \ 
    --pixel \
    --proj-file /path/to/projections.mrc \
    --angle-file /path/to/angles.tlt \
    --save-dir /path/to/save/directory/ \
    --save-name output_volume.mrc \
    --device 0 \
    --n3 256 \ # The size of the volume along the z-axis, you can modify this based on your data
    --batch-size 100000  # Depends on the memory of your GPU
```

## Legacy interface (v1)
The legacy `super.py` and `super-list.py` interface documentation has moved to [`docs/v1-interface.md`](docs/v1-interface.md).
