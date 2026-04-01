## Training your own model
This section provides instructions for training your own model using CryoLithe. We provide a sample training configuration file which runs training on 4 tomograms from the EMPIAR-11830 dataset. You can modify this configuration file to train on your own data.

### 1. Prepare your training data
CryoLithe's training pipeline expects the following data for each tomogram:
- Reference tomograms (e.g. denoised tomograms). These are used as the "ground truth" for training. You can generate these using your preferred denoising methods such as Icecream, Isonet,
 DeepDeWedge, cryoCARE,  or any other methods of your choice.
- Tilt-series projection data. For this you can use dose-weighted, CTF-corrected or uncorrected projections. In the current training pipeline, cryolithe can also use ODD/EVN separated projections for better training. If you don't have ODD/EVN separated projections, you can just provide the projections and set the ODD/EVN paths to YAML `null` (or leave them as an empty list) in the config file.
- Tilt angles corresponding to the tilt-series projections. These should be in .tlt format (a text file with one angle per line).
- (Optional) z_lims_list for each tomogram, which specifies the z-slices to be used for training. If not provided we use a global limits in the training config file.

We provide a command line to download a small subset of the EMPIAR-11830 dataset for quick testing:
```bash
cryolithe download-training-data --small-subset
```
You can also download the full dataset using the same command without the `--small-subset` flag, but please note that the full dataset is quite large (600+ GB) and will require significant storage space and time to download.

### 2. Create a training configuration file
Next, you need to create a YAML configuration file that specifies the training parameters and paths to your training data. You can use the provided `sample_model_training.yaml` file as a template. Make sure to update the paths to your training data and adjust any training parameters as needed. 

More detailed training config options are present in `train_model.yaml` present in src/CryoLithe/.

### 3. Run the training command
Once you have your training data prepared and your configuration file ready, you can run the training command using CryoLithe's CLI. Use the following command, replacing the path to the config file with your own:
```bash
cryolithe train-model --config path/to/your/sample_model_training.yaml
```
This will start the training process. The model checkpoints and training logs will be saved in the output directory specified in your training config file.

Optionally, if your weights and biases (wandb) logging is enabled in the training config, you can monitor the training progress and metrics in real-time on your wandb dashboard.

The config file 'sample_model_training.yaml' provides a simple example of training a pixel-wise model on 4 tomograms from the EMPIAR-11830 dataset. You can modify this config file to train on your own data and adjust the training parameters as needed.
An example of a detailed training config file is provided in `src/CryoLithe/train_model.yaml`. All the possible parameters are described in the following paragraph.

#### Training the pixel-wise model
To train the pixel-wise model set `use_wavelet_trainer` parameter to false in the config file. 


#### Pre-trained model
To train a model close to ones provided, you can use the yaml file 'model_training.yaml' present in the docs directory and run:
```bash
cryolithe train-model --config model_training.yaml
```

### 4- test the trained model
After training, you can test the trained model on new projections using the `cryolithe reconstruct` command as described in the previous section. Make sure to specify the path to your trained model directory when running the reconstruct command.
```bash
cryolithe reconstruct \
    --model-dir ./training-run/sample/ \
    --proj-file ./cryolithe-training-data/empiar-11830/tomo_005/proj_CTF.mrc \
    --angle-file ./cryolithe-training-data/empiar-11830/tomo_005/10092023_NNPK_Arctis_WebUI_Ron_grid8_Position_2.rawtlt \
    --save-dir ./training-run/ \
    --save-name cryolithe_training_example.mrc \
    --device 0 \
    --n3 512 \
    --batch-size 100000  # Depends on the memory of your GPU
```

## Training parameters
There is a large number of parameters that can be adjusted for training. These are specified in the YAML config file used for training. Below is a detailed description of the parameters that can be set in the training config file.
We first report the most critical ones that we recommend adjusting, if the default is not working, when training on a new dataset. We then provide a comprehensive list of all parameters that can be set in the config file.

### Most critical parameters to adjust for training on a new dataset
| Variable                      | Default | Description |
|-------------------------------|------------|---|
| `output_dir`                  | `null` | Directory to save outputs such as models and logs. |
| `device`                      | `0` | CUDA device id. |
| **dataset**                   |                                       ||
| `dataset.root_dir`            | `null`                                | Root directory of the dataset. Should contain subdirectories for each sample, with projections and angles. |
| **splits**                    |                                       ||
| `splits.mode`                 | `ratio`                               | Data splitting mode. Options: `ratio` or `explicit`. |
| `splits.train`                | `0.8`                                 | Proportion of data used for training. |
| `splits.valid`                | `0.1`                                 | Proportion of data used for validation. |
| `splits.test`                 | `0.1`                                 | Proportion of data used for testing. Train, valid, and test should sum to 1. |
| **model**                     |                                       ||
| `model.patch_size`            | `21`                                  | Patch size around each point. Should be odd to define a central pixel. Effective field of view is `patch_size * patch_scale`. |
| **training**                  |                                       ||
| `training.batch_size`         | `1`                                   | Training batch size. |
| `training.batch_size_volume`  | `1`                                   | Batch size used by the volumetric trainer. |
| `training.num_epochs`         | `10000`                               | Number of training epochs. |
| `training.num_points`         | `15000`                               | Number of sampled points from the volume when using the volumetric trainer. |
| `training.lr`                 | `0.0001`                              | Learning rate. |
| `training.save_every`         | `50`                                  | Save the model every this many epochs. |
| `training.display_every`      | `200`                                 | Display training progress every this many epochs. |
| `training.downsample`         | `false`                               | Whether to downsample projections and volume for training. |
| `training.downsample_factor`  | `[0.5, 0.25]`                         | Downsampling factors used if `downsample` is enabled. |
| `training.use_half_precision` | `false`                               | Whether to use float16 training. |
| `training.preload`            | `true`                                | Whether to preload the dataset into memory before training. |

### Other parameters
| Variable             | Default                               | Description |
|----------------------|---------------------------------------|---|
| `seed`               | `42`                                  | Random seed for reproducibility. |
| `filter_projections` | `true`                                | Whether to apply a filter to projections before feeding them to the model. |
| `use_2D_filters`     | `false`                               | Whether to apply 2D filters to projections. |
| **data**             |                                       ||
| `data.type`          | `t20`                                 | Dataset type, for example `t20` or `empiar10185`. |
| `data.n_projections` | `null`                                | Number of projections to use from the dataset. If `null`, all available projections are used. |
| **dataset**             |                                       ||
| `dataset.normalize_type` | `standard`                            | Normalization type applied to the projections. Options: `standard`, `minmax`, `none`. |
| `dataset.randomize_projections` | `true`                                | Randomly select, if available, between odd, even, or unsplit tilt-series. If `false`, use the full tilt-series. |
| `dataset.full_randomize` | `true`                                | If `true`, select uniformly between odd, even, full tilt-series, or a mix of odd and even projections. If `false`, select between odd, even, and full only. |
| `dataset.remove_projections_wedge` | `false`                               | If `true`, randomly remove projections near the missing wedge. |
| `dataset.max_remove_projections_wedge` | `30`                                  | Maximum number of projections to remove near the wedge if `remove_projections_wedge` is enabled. |
| `dataset.remove_projections_random` | `true`                                | If `true`, randomly remove projections from the tilt-series. |
| `dataset.max_remove_projections_random` | `8`                                   | Maximum number of randomly removed projections if `remove_projections_random` is enabled. |
| `dataset.vol_paths` | `[]`                                  | List of volume file paths. |
| `dataset.projection_paths` | `[]`                                  | List of projection file paths. |
| `dataset.projection_paths_odd` | `[]`                                  | List of odd projection file paths. Optional. |
| `dataset.projection_paths_even` | `[]`                                  | List of even projection file paths. Optional. |
| `dataset.angle_paths` | `[]`                                  | List of angle file paths. |
| `dataset.z_lims_list` | `[]`                                  | List of z-limits. |
| **splits**             |                                       ||
| `splits.shuffle` | `true`                                | Whether to shuffle the data before splitting. |
| `splits.seed` | `0`                                   | Random seed used for shuffling before splitting. |
| `splits.explicit.train` | `[]`                                  | Explicit list of training indices if `splits.mode` is `explicit`. |
| `splits.explicit.valid` | `[]`                                  | Explicit list of validation indices if `splits.mode` is `explicit`. |
| `splits.explicit.test` | `[]`                                  | Explicit list of test indices if `splits.mode` is `explicit`. |
| `splits.cache.train` | `true`                                | Whether to cache training data in memory. |
| `splits.cache.valid` | `true`                                | Whether to cache validation data in memory. |
| `splits.cache.test` | `false`                               | Whether to cache test data in memory. |
| **model**             |                                       ||
| `model.type` | `SliceSet`                            | Model type. Only `SliceSet` is implemented so far. |
| `model.mlp_output` | `32`                                  | Output size of the MLP processing each slice before combination. |
| `model.slice_index` | `2`                                   | Index of the slice used as model input. |
| `model.compare_index` | `3`                                   | Index of the slice used for comparison in the loss. Must differ from `slice_index`. |
| `model.learn_residual` | `false`                               | If `true`, learn the residual between input and target slice instead of predicting the target directly. |
| `model.set_input` | `64`                                  | Input size of the set-processing MLP. |
| `model.set_output` | `128`                                 | Output size of the set-processing MLP. |
| `model.set_num_layers` | `1`                                   | Number of layers in the set-processing MLP. |
| `model.set_hidden_size` | `128`                                 | Hidden size of the set-processing MLP. |
| `model.set_skip_connection` | `true`                                | Whether to use skip connections in the set-processing MLP. |
| `model.set_bias` | `false`                               | Whether to use bias in the set-processing MLP. |
| `model.radon_bias` | `false`                               | Whether to use bias in the Radon transform layer. |
| `model.learned_positional_encoding` | `false`                               | Whether to use learned positional encoding for slices. |
| `model.learned_positional_encoding_use_softmax` | `true`                                | Whether to normalize learned positional encodings with softmax. |
| `model.slice_transformer_avg_pooling` | `true`                                | Whether to apply average pooling to the slice transformer output before combination. |
| `model.slice_transformer_positional_encoding_size` | `64`                                  | Positional encoding size for the slice transformer. |
| `model.sice_transformer_transformer_positional_encoding_add_angle` | `false`                               | Whether to add angle information to the slice transformer positional encoding. |
| `model.sice_transformer_positional_encoding_mult_angle` | `true`                                | Whether to multiply angle information with the slice transformer positional encoding. |
| `model.slice_mlp_hidden_size` | `512`                                 | Hidden size of the slice-processing MLP. |
| `model.slice_mlp_num_layers` | `5`                                   | Number of layers in the slice-processing MLP. |
| `model.slice_mlp_skip_connection` | `true`                                | Whether to use skip connections in the slice-processing MLP. |
| `model.slice_mlp_bias` | `false`                               | Whether to use bias in the slice-processing MLP. |
| `model.combine_mlp_layers` | `5`                                   | Number of layers in the MLP that combines slices. |
| `model.combine_mlp_hidden` | `512`                                 | Hidden size of the MLP that combines slices. |
| `model.combine_dropout` | `0`                                   | Dropout rate for the combine MLP. |
| `model.combine_batch_norm` | `false`                               | Whether to use batch normalization in the combine MLP. |
| `model.combine_learn_residual` | `false`                               | Whether the combine MLP learns a residual instead of a direct prediction. |
| `model.combine_skip_connection` | `true`                                | Whether to use skip connections in the combine MLP. |
| `model.combine_mlp_bias` | `false`                               | Whether to use bias in the combine MLP. |
| `model.output_size` | `8`                                   | Output size of the model. |
| **ramp**             |                                       ||
| `ramp.filter_type` | `cosine`                              | Type of ramp filter applied to projections. Options: `cosine`, `linear`, `learnable`. |
| `ramp.use_learnable_ramp` | `false`                               | Whether to use a learnable ramp filter. |
| `ramp.use_pretrained` | `false`                               | Whether to initialize the ramp filter from pretrained weights. |
| `ramp.pretrain_path` | `null`                                | Path to pretrained ramp filter weights. |
| `ramp.type` | `VectorModel_real`                    | Type of learnable ramp filter. Options: `VectorModel_real`, `VectorModel_complex`, `FIR`. |
| `ramp.model.init` | `cosine`                              | Initialization method for the learnable ramp filter. Options: `cosine`, `linear`, `impulse`, `random`. |
| `ramp.model.size` | `600`                                 | Size of the learnable ramp filter. |
| **filter_2D**             |                                       ||
| `filter_2D.type` | `fir`                                 | Type of 2D filter. Options: `fir`, `learnable`. |
| `filter_2D.model.init` | `impulse`                             | Initialization method for the 2D filter. Options: `impulse`, `random`. |
| `filter_2D.model.size` | `101`                                 | Size of the 2D filter. Should be odd to define a central pixel. |
| **training**             |                                       ||
| `training.use_volumetric_trainer` | `false`                               | Whether to use a volumetric trainer that samples points from the volume. |
| `training.use_wavelet_trainer` | `true`                                | Whether to use a wavelet-based trainer on projections. |
| `training.lr_scheduler_type` | `linear_warmup_cosine_annealing`      | Learning-rate scheduler type. Options include `linear_warmup_cosine_annealing`, `cosine_annealing`, `step`, `none`. |
| `training.lr_warmup_epochs` | `500`                                 | Number of warmup epochs for the scheduler. |
| `training.lr_min` | `0.0000001`                           | Minimum learning rate for cosine annealing schedules. |
| `training.weight_decay` | `0.00001`                             | Weight decay for the optimizer. |
| `training.ramp_weight_decay` | `0`                                   | Weight decay applied to the learnable ramp filter. |
| `training.optimizer` | `adamw`                               | Optimizer type. Options: `adam`, `adamw`, `sgd`. |
| `training.loss` | `l1`                                  | Loss function. Options: `l1`, `l2`, `smooth_l1`. |
| `training.learn_patch_scale` | `false`                               | Whether to learn the patch scale during training. |
| `training.patch_scale_init` | `1`                                   | Initial patch scale. |
| `training.normalize_patch` | `false`                               | Whether to normalize patches before feeding them to the model. |
| `training.patch_ord` | `inf`                                 | Ordering rule for patch pixels. |
| `training.use_angle` | `true`                                | Whether to include projection angle information as model input. |
| `training.boundary` | `false`                               | Whether to include a boundary loss. |
| `training.discrete_sampling` | `true`                                | Whether to sample volume points discretely. |
| `training.get_projection_prefiltered` | `false`                               | Whether to use projections prefiltered by the chosen ramp filter. |
| `training.scale_data` | `false`                               | Whether to scale the input data to a specific range. |
| `training.load_volume_epoch` | `20`                                  | Epoch from which the volume starts being used in training. |
| `training.generate_projection_epoch` | `10`                                  | Epoch from which projections are generated from the predicted volume during training. |
| `training.repeat` | `1`                                   | Number of times to repeat the training loop. |
| `training.downsample_proj_vol` | `false`                               | Whether to downsample both projections and volume. |
| `training.downsample_proj_vol_factor` | `0.5`                                 | Downsampling factor for projections and volume. |
| `training.nlims` | `[[100, 900], [80, 900], [200, 300]]` | Limits for random sampling of points from the volume along x, y, and z. |
| `training.dynamic_zlims` | `false`                               | Whether to dynamically adjust z-limits during training. |
| `training.use_amp` | `true`                                | Whether to use automatic mixed precision. |
| `training.wavelet` | `bior3.1`                             | Wavelet type for the wavelet-based trainer. |
| `training.wavelet_levels` | `1`                                   | Number of wavelet decomposition levels. |


[Back to README](../README.md) <br>
[Previous: Reconstructing the tomograms using the trained models](./4-example_reconstruction.md) <br>

