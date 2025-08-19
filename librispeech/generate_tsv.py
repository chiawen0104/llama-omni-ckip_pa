import json
import os
import soundfile

json_dir = "data/"
output_root = "data/"
splits = ["dev", "test", "train"]

for split in splits:
    json_path = os.path.join(json_dir, f"{split}.json")
    tsv_path = os.path.join(output_root, f"{split}.tsv")

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(tsv_path, "w") as f:
        f.write("librispeech\n")
        for item in data:
            file_id = item["id"]
            speaker_id, chapter_id, _ = file_id.split('-')
            path = f"LibriSpeech/{split}-clean/{speaker_id}/{chapter_id}/{file_id}.flac"
            nsample = soundfile.info(path).frames
            f.write(f"{path}\t{nsample}\n")

    print(f"Wrote {tsv_path}")
