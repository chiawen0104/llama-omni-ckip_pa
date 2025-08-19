import json
import os

def process_librispeech_data(root_dir, output_file):
    """
    Processes Librispeech dev-clean data and formats it into a JSON file.

    Args:
        root_dir (str): The root directory containing the dev-clean data.
        output_file (str): The path to the output JSON file.
    """
    data = []

    # The file structure is assumed to be `root_dir/speaker_id/chapter_id/`
    for speaker_id in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_id)

        # Check if it's a directory
        if not os.path.isdir(speaker_path):
            continue

        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            
            # Check if it's a directory
            if not os.path.isdir(chapter_path):
                continue
            
            # Check for the .trans.txt file
            trans_file_path = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            if not os.path.exists(trans_file_path):
                continue

            with open(trans_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) < 2:
                        continue

                    transcript_id, transcript_text = parts
                    
                    # Construct the relative path to the .flac file
                    # The path should be relative to the dataset root or a logical base.
                    # This example uses a common structure seen in LibriSpeech datasets.
                    # flac_path = os.path.join("LibriSpeech", "dev-clean", speaker_id, chapter_id, f"{transcript_id}.flac")
                    flac_path = "librispeech/tts_instruct.wav"
                    item = {
                        "id": transcript_id,
                        "speech": flac_path,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<speech>\nScript: {transcript_text}"
                            },
                            {
                                "from": "assistant",
                                "value": transcript_text
                            }
                        ]
                    }
                    data.append(item)
    print(f"len: {len(data)}")
    # Write the data to the JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # Replace 'path/to/your/dev-clean' with the actual path to your 'dev-clean' folder.
    # The image suggests a path like 'librispeech/Librispeech/dev-clean'.
    file_path = "LibriSpeech/train-clean"
    output_json_path = "data/train.json"

    process_librispeech_data(file_path, output_json_path)
    print(f"Successfully generated {output_json_path}")