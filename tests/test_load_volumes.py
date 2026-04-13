
import os
import numpy as np
import torch
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path


save_dir = "./tests/png/"

# Create folder if not exists
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Load trained volume and display some images
vol = mrcfile.open("./training-run/cryolithe_training_example.mrc", permissive=True).data.copy()
for i in range(0, vol.shape[0], 10):
    plt.imshow(vol[i], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"vol_example_custom_trained_slice_{str(i).zfill(5)}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

vol = mrcfile.open("./results/sample_reconstruction/vol_sample_pixel.mrc", permissive=True).data.copy()
for i in range(0, vol.shape[0], 10):
    plt.imshow(vol[i], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"vol_example_cryolithe_slice_{str(i).zfill(5)}.png"), bbox_inches='tight',
                pad_inches=0)
    plt.close()

