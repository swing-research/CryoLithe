# CryoLithe: Rapid Cryo-ET Reconstruction via Transform-Localized Deep Learning 	
Official repo for CryoLithe ([paper](https://arxiv.org/abs/2501.15246), [website](https://valentindebarnot.github.io/cryolithe.html))

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
- 01.04.2026:
    - Update readme.
    - Add training procedure.
- 19.02.2026:
    - Command line interface + uv installation (super fast) + trained model in Hugging Face.
- 05.09.2025
    - New model trained on a larger dataset.
- 05.08.2025
    - Support list of volumes. 
- 18.04.2025:  
    - Arbitrary nummber of tilts + recommended pytorch version to 2.6.0 + Multi-GPU inference

## My Section Title

## Overview


[Rapid installation, usage and reconstruction](#rapid-installation-usage-and-reconstruction)

[Installation instructions](docs/1-installation.md)

[Download the trained models](docs/2-download_trained_models.md)

[Download examples](docs/3-download_examples.md)

[Reconstructing the tomograms using the trained models](docs/4-example_reconstruction.md)

[Training CryoLithe on your own data](docs/5-training.md)

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




## Legacy interface (v1)
The legacy `super.py` and `super-list.py` interface documentation has moved to [`docs/v1-interface.md`](docs/v1-interface.md).
