import subprocess
from pathlib import Path

# Use first
# conda activate CryoLithe

tests = [
    {
        "name": "Test 1 - downlaod model",
        "cmd": ["cryolithe", "download", "--local-dir", "./trained_models/"],
    },
    {
        "name": "Test 2 - download model with no override",
        "cmd": ["cryolithe", "download", "--local-dir", "./trained_models/", "--no-override-model-dir"],
    },
    {
        "name": "Test 3 - download sample data",
        "cmd": ["cryolithe", "download-sample-data"],
    },
    {
        "name": "Test 4 - download sample data with no override",
        "cmd": ["cryolithe", "download-sample-data", "--no-override-data"],
    },
    {
        "name": "Test 5 - download training data (small subset)",
        "cmd": ["cryolithe", "download-training-data", "--small-subset", "--local-dir", "./cryolithe-training-data"],
    },
    {
        "name": "Test 6 - download training data (small subset) with no override",
        "cmd": ["cryolithe", "download-training-data", "--small-subset", "--local-dir", "./cryolithe-training-data", "--no-override-data"],
    },
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