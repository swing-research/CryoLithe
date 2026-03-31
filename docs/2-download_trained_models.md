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


[Back to README](../README.md) <br>
[Previous: Installation](./1-installation.md) <br>
[Next: Download examples](./3-download_examples.md)