# End-to-end localized deep learning for Cryo-ET 	
Official repo for CryoLithe ([paper](https://arxiv.org/abs/2501.15246))

**Abstract**
Cryo-electron tomography (cryo-ET) enables 3D visualization of cellular structures. 
Accurate reconstruction of high-resolution volumes is complicated by the very low signal-to-noise ratio and a restricted range of sample tilts. 
Recent self-supervised deep learning approaches, which post-process initial reconstructions by filtered backprojection (FBP), have significantly improved reconstruction quality with respect to signal processing iterative algorithms, but they are slow, taking dozens of hours for an expert to reconstruct a tomogram and demand large memory. 
We present CryoLithe, an end-to-end network that directly estimates the volume from an aligned tilt series. 
CryoLithe achieves denoising and missing wedge correction comparable or better than state-of-the-art self-supervised deep learning approaches such as [Icecream](https://github.com/swing-research/icecream), [Cryo-CARE](https://github.com/juglab/cryoCARE_pip), [IsoNet](https://github.com/IsoNet-cryoET/IsoNet) or [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge), while being two orders of magnitude faster. 
To achieve this, we implement a local, memory-efficient reconstruction network.
We demonstrate that leveraging transform-domain locality makes our network robust to distribution shifts, enabling effective supervised training and giving excellent results on real data---without retraining or fine-tuning.
CryoLithe reconstructions facilitate downstream cryo-ET analysis, including segmentation and subtomogram averaging and is openly available.



## Updates 
- XX.XX.2026:
    - Update readme.  
- 19.02.2026:
    - Command line interface + uv installation (super fast) + trained model in Hugging Face.
- 05.09.2025
    - New model trained on a larger dataset.
- 05.08.2025
    - Support list of volumes. 
- 18.04.2025:  
    - Arbitrary nummber of tilts + recommended pytorch version to 2.6.0 + Multi-GPU inference

## Rapid installation, usage and reconstruction
The first goal of CryoLithe is to be ultra-fast. Thus, we have made the installation and usage as simple as possible.
For custom usage, please see the next sections. 

**Installation**
```bash
conda create -n CryoLithe python=3.9
conda activate CryoLithe
pip3 install torch torchvision torchaudio
pip install git+https://github.com:swing-research/CryoLithe.git
```

**Download the trained models**
```bash
cryolithe download
```

**Reconstruction**
```bash
cryolithe reconstruct \ 
    --proj-file path_projections \
    --angle-file path_angles \
    --save-dir ./CryoLithe_results/ \
    --save-name output_volume.mrc \
    --device 0 \
    --n3 256 \ 
    --batch-size 100000  # Depends on the memory of your GPU
```
where `path_projections` and `path_angles` are the paths to the projection (mrc or st file) and angle files (tlt or txt) of your tilt-series.

## Installation
You can install CryoLithe using either [**conda**](#using-conda-recommended-if-you-already-use-conda) (recommended if you already use conda) or [**uv**](#using-uv) (lightning fast installer).

### Using conda (recommended if you already use conda):

Create a new conda environment using the command:
```bash
conda create -n CryoLithe python=3.9
conda activate CryoLithe
pip3 install torch torchvision torchaudio
```
The code was tested with PyTorch 2.6 and 2.8.

You can install CryoLithe and its dependencies directly from the GitHub repository using pip or clone the repository and install it locally.
####  Direct installation (recommended):
```bash
pip install git+https://github.com:swing-research/CryoLithe.git
```

#### Local installation (for ones who want to modify the code):
Clone the repository using **one** of the following methods:

*HTTPS*
```bash
git clone https://github.com:swing-research/CryoLithe.git
```
*SSH (for users with SSH keys configured):*
```bash
git clone git@github.com:swing-research/CryoLithe.git
```

Install in editable mode:
```bash
pip install -e .
```

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

To uninstall the tool, run:
```bash
uv tool uninstall cryolithe
```

### Test the installation
Run: 
```bash
cryolithe --help
```
It should display the main commands: `reconstruct`, `download`, and `download-sample-data`


## Download the trained models
The trained models are stored in Hugging Face and can be downloaded using the `cryolithe download` command. You can specify the local directory where you want to save the models using the `--local-dir` flag. For example, to save the models in a directory called `models`, run:

```bash
cryolithe download --local-dir /path/to/trained/model/
```
If you want it to be saved in the Hugging Face cache, then run:
```bash
cryolithe download
```

Note that this will create a file called .cryolithe.yaml in your home directory with the path to the models files. 
This is used by the reconstruct command to load models when you don't specify the path to the model directory. 
By default, each new `cryolithe download` updates `model_dir` in this path to the newly downloaded model file. 
If you want to keep the existing files, use:
```bash
cryolithe download --no-override-model-dir
```

Currently, we provide two models:
 - 'cryolithe_pixel' - trained to recover the volume one pixel at a time (8x longer, top quality)
 - 'cryolithe' - trained to recover the wavelet coefficients of the volume (faster, small loss of quality).

## Download examples (optional)
Download an example of a tilt-series (73mb) of ribosomes from Hugging Face dataset repo:
```bash
cryolithe download-sample-data
```
By default, this downloads to `./cryolithe-sample-data` in the directory where you run the command. You can also provide your own path:
```bash
cryolithe download-sample-data --local-dir /path/to/data
```


## Reconstructing the tomograms using the trained models
Once the models are downloaded, you can use them to reconstruct the tomograms from your own tilt-series. 
This is the `reconstruct` command.
Specific arguments can be provided in a yaml file or directly in the command line.
An example yaml file is provided in the repository as [`ribo80.yaml`](ribo80.yaml) which contains the necessary information to run the model on the example ribosome dataset.

**Note**: Make sure to update the paths in the yaml file to point to the correct locations of the projection and angle files and model directory on your system before running the command.

```bash
cryolithe reconstruct --config ribo80.yaml
```

Alternatively, a convenient way would be to use the command line arguments to run the model. 
For example, to run the model on a sample dataset, you can use the following command:
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
All the parameters and their descriptions are available using the following command:
```bash
cryolithe reconstruct --help
```
Most paramaters have default value, you can simply run the command with the necessary parameters and it will use the default values for the rest. 
For example, if you have downloaded the models using the `cryolithe download` command, you can simply run:
```bash
cryolithe reconstruct \ 
    --proj-file /path/to/projections.mrc \
    --angle-file /path/to/angles.tlt \
    --save-dir /path/to/save/directory/ \
    --save-name output_volume.mrc \
    --device 0 \
    --n3 256 \ # The size of the volume along the z-axis, you can modify this based on your data
    --batch-size 100000  # Depends on the memory of your GPU
```

If the model directory is not provided, the reconstruct code will choose the wavelet model by default. 
If you want to use the pixel model, you can use the flag `--pixel` to specify that you want to use the pixel model.

## Legacy interface (v1)
The legacy `super.py` and `super-list.py` interface documentation has moved to [`docs/v1-interface.md`](docs/v1-interface.md).
