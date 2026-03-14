from huggingface_hub import hf_hub_download
import os

token = os.getenv("HF_TOKEN")
for f in ["config.json", "modeling_deepfake.py", "processor_deepfake.py", "inference.py"]:
    path = hf_hub_download("Naman712/Deep-fake-detection", f, token=token)
    print(f"=== {f} ===")
    print(open(path).read())
    print()
