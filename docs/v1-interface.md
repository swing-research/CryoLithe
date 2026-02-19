# CryoLithe Legacy Interface (v1)

This document preserves the older inference workflow based on `super.py` and `super-list.py`.

## Running the model

The script `super.py` is used to run the trained model on new projections. The script requires a YAML configuration file with the following fields:
- `model_dir` - path to the directory containing the trained model
- `proj_file` - path to the projection file
- `angle_file` - path to the angle file
- `save_dir` - path to the directory where the output will be saved
- `save_name` - name of the output volume
- `device` - CUDA device id to run the model on (integer, e.g. `0`; older code also supports a list of ids, e.g. `[0, 1]`)
- `downsample_projections` - whether to downsample the projections
- `downsample_factor` - factor by which to downsample the volume
- `anti_alias` - whether to apply anti-aliasing to the projections
- `N3` - size of the volume along the z-axis
- `batch_size` - batch size to use when running the model

Run the script:

```bash
python3 super.py --config <path_to_config_file>
```

A sample YAML file is provided as `ribo80.yaml`.

## Running the model on the ribosome dataset

Download the ribosome dataset:

```bash
bash download_ribosome.sh
```

This places data in the `data` directory. The dataset includes:
- `projections.mrcs` - the projections
- `angles.tlt` - the angles

To run:

```bash
python3 super.py --config ribo80.yaml
```

## Using the wavelet model

```bash
python3 super.py --config ribo80_wavelet.yaml
```

## Running on a list of projections

Use `super-list.py` with `ribo80_list.yaml`. You can set multiple values for:
- `proj_file` - list of projection paths
- `angle_file` - list of angle paths
- `save_name` - list of output names
- `N3` - list of z-axis sizes

Run:

```bash
python3 super-list.py --config ribo80_list.yaml
```

## Downloading older models

```bash
bash download_old.sh
```
