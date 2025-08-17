import json
import os
import soundfile

json_dir = "."
output_root = "."
splits = ["train", "valid", "test"]

for split in splits:
    json_path = os.path.join(json_dir, f"{split}.json")
    tsv_path = os.path.join(output_root, f"{split}.tsv")

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(tsv_path, "w") as f:
        f.write("speechocean\n")
        for item in data:
            path = os.path.relpath(item["speech"], start="speechocean")
            nsample = soundfile.info(path).frames  # 音檔總長度（frame數）
            f.write(f"{path}\t{nsample}\n")

    print(f"Wrote {tsv_path}")
