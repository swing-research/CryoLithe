# Training your own model
This section provides instructions for training your own model using CryoLithe. We provide a sample training configuration file which runs training on 4 tomograms from the EMPIAR-11830 dataset. You can modify this configuration file to train on your own data.

## 1. Prepare your training data
CryoLithe's training pipeline expects the following data for each tomogram:
- Reference tomograms (e.g. denoised tomograms). These are used as the "ground truth" for training. You can generate these using your preferred denoising methods such as Icecream, Isonet,
 DeepDeWedge, cryoCARE,  or any other methods of your choice.
- Tilt-series projection data. For this you can use dose-weighted, CTF-corrected or uncorrected projections. In the current training pipeline, cryolithe can also use ODD/EVN separated projections for better training. If you don't have ODD/EVN separated projections, you can just provide the projections and set the ODD/EVN paths to YAML `null` (or leave them as an empty list) in the config file.
- Tilt angles corresponding to the tilt-series projections. These should be in .tlt format (a text file with one angle per line).
- (Optional) z_lims_list for each tomogram, which specifies the z-slices to be used for training. If not provided we use a global limits in the training config file.

Additionally, we provide additional options in the command line to download a small subset of the EMPIAR-11830 dataset for quick testing. You can use the following command to download the data:
```bash
cryolithe download-training-data --small-subset
```
You can also download the full dataset using the same command without the `--small-subset` flag, but please note that the full dataset is quite large (600+ GB) and will require significant storage space and time to download.

You can directly use the subset data with the provided `sample_model_training.yaml` config file by setting the `root_dir` to the path where the data is downloaded and train the model.

## 2. Create a training configuration file
Next, you need to create a YAML configuration file that specifies the training parameters and paths to your training data. You can use the provided `sample_model_training.yaml` file as a template. Make sure to update the paths to your training data and adjust any training parameters as needed. 

More detailed training config options are present in `train_model.yaml` present in src/CryoLithe/.

## 3. Run the training command
Once you have your training data prepared and your configuration file ready, you can run the training command using CryoLithe's CLI. Use the following command, replacing the path to the config file with your own:
```bash
cryolithe train-model --config path/to/your/training_config.yaml
```

This will start the training process. The model checkpoints and training logs will be saved in the output directory specified in your training config file.

Optionally, if you weights and biases (wandb) logging is enabled in the training config, you can monitor the training progress and metrics in real-time on your wandb dashboard.


### Training the pixel-wise model
To train the pixel-wise model set `use_wavelet_trainer`  parameter to false in the config file. The yaml file will look something like this:
```yaml
training:
    use_wavelet_trainer: false
```

