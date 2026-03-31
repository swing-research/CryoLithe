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

