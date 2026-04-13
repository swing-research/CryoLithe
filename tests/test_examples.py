import subprocess
from pathlib import Path
# Use first
# conda activate CryoLithe

tests = [
    {
        "name": "Test 1 - reconstruct from yaml file",
        "cmd": ["cryolithe", "reconstruct", "--config", "docs/ribo80.yaml"],
    },
    {
        "name": "Test 2 - reconstruct with command-line args",
        "cmd": [
            "cryolithe",
            "reconstruct",
            "--model-dir",
            "./trained_models/cryolithe-pixel/",
            "--proj-file",
            "./cryolithe-sample-data/projections.mrc",
            "--angle-file",
            "./cryolithe-sample-data/angles.tlt",
            "--save-dir",
            "./results/sample_reconstruction/",
            "--save-name",
            "./vol_sample_pixel.mrc",
            "--device",
            "0",
            "--n3",
            "100",
            "--batch-size",
            "50000",
            "--downsample-projections",
            "--downsample-factor",
            "0.25",
        ],
    },
    {
        "name": "Test 3 - reconstruct from yaml file with volumes as list",
        "cmd": ["cryolithe", "reconstruct", "--config", "docs/ribo80_list.yaml"],
    },
    {
        "name": "Test 4 - reconstruct from yaml file with wavelet model",
        "cmd": ["cryolithe", "reconstruct", "--config", "docs/ribo80.yaml"],
    },
    {
        "name": "Test 5 - train model with yaml file",
        "cmd": ["cryolithe", "train-model", "--config", "docs/sample_model_training.yaml"],
    },
    {
        "name": "Test 6 - reconstruct from trained model",
        "cmd": ["cryolithe", "reconstruct",
            "--model-dir", "./training-run/sample/",
            "--proj-file", "./cryolithe-training-data/empiar-11830/tomo_005/proj_CTF.mrc",
            "--angle-file", "./cryolithe-training-data/empiar-11830/tomo_005/10092023_NNPK_Arctis_WebUI_Ron_grid8_Position_2.rawtlt",
            "--save-dir", "./training-run/",
            "--save-name", "cryolithe_training_example.mrc",
            "--device", "0",
            "--n3", "256",
            "--batch-size", "100000",
            "--downsample-projections", "--downsample-factor", "0.5",
                ],
    },
    {
        "name": "Test 7 - Save png to check manually",
        "cmd": ["python", "./tests/test_load_volumes.py",]
    }
]

log_dir = Path("test_logs")
log_dir.mkdir(exist_ok=True)

failed = []

for i, test in enumerate(tests, start=1):
    name = test["name"]
    cmd = test["cmd"]
    log_file = log_dir / f"test_{i:02d}.log"

    print(f"Running: {name}")
    print("Command:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_file.write_text(result.stdout)

    if result.returncode != 0:
        failed.append(
            {
                "name": name,
                "cmd": cmd,
                "returncode": result.returncode,
                "log": str(log_file),
            }
        )
        print(f"FAILED: {name}")
    else:
        print(f"PASSED: {name}")

    print("-" * 60)

print("\n=== FINAL SUMMARY ===")

if not failed:
    print("All tests passed.")
else:
    print(f"{len(failed)} test(s) failed:\n")
    for item in failed:
        print(f"- {item['name']}")
        print(f"  Command: {' '.join(item['cmd'])}")
        print(f"  Return code: {item['returncode']}")
        print(f"  Log file: {item['log']}")
        print()