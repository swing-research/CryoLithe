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


[Back to README](../README.md) <br>
[Previous: Download examples](./3-download_examples.md) <br>
[Next: Training your own model](./5-training.md)