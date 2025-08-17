import json
from pathlib import Path
import random
import re

def load_text_file(path):
    with open(path, "r") as f:
        return dict(line.strip().split(maxsplit=1) for line in f if line.strip())


def build_dataset(split_dir: Path, scores: dict):
    wav_scp = load_text_file(split_dir / "wav.scp")
    text = load_text_file(split_dir / "text")

    dataset = []
    for utt_id, wav_path in wav_scp.items():
        if utt_id not in scores:
            continue  # skip if no score
        if utt_id not in text:
            continue  # skip if no text

        score = scores[utt_id]
        prompt = f"<speech>\nPlease assess the speaker's pronunciation of the following script:\n{score['text']}\nScore the accuracy, completeness, fluency, prosody, and total at the sentence level. Each score should range from 0 to 10."
        result = f"accuracy: {score['accuracy']}, completeness: {score['completeness']}, fluency: {score['fluency']}, prosodic: {score['prosodic']}, total: {score['total']}"
        base_dir = 'speechocean'
        entry = {
            "id": utt_id,
            "speech": base_dir + '/' + wav_path,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": result}
            ]
        }
        dataset.append(entry)

    return dataset


def split_train_valid_set(dataset):
    complete_10 = []
    complete_not_10 = []

    for item in dataset:
        score_str = item["conversations"][-1]["value"]
        match = re.findall(r'completeness:\s*([0-9]+)', score_str)
        if match:
            completeness = int(match[0])
            if completeness == 10:
                complete_10.append(item)
            else:
                complete_not_10.append(item)

    # 打亂順序
    random.shuffle(complete_10)
    random.shuffle(complete_not_10)

    # 設定比例，80% 訓練、20% 驗證
    train_10 = complete_10[:int(0.8 * len(complete_10))]
    valid_10 = complete_10[int(0.8 * len(complete_10)):]

    train_not_10 = complete_not_10[:int(0.8 * len(complete_not_10))]
    valid_not_10 = complete_not_10[int(0.8 * len(complete_not_10)):]

    # 合併
    train = train_10 + train_not_10
    valid = valid_10 + valid_not_10

    # 最後再隨機打亂一次
    random.shuffle(train)
    random.shuffle(valid)

    return train, valid


def main():
    root = Path(__file__).parent.resolve()
    scores_path = root / "resource" / "scores.json"
    scores = json.load(open(scores_path, "r"))
    

    train, valid, test = [], [], []
    for split in ["train", "test"]:
        split_dir = root / split
        dataset = build_dataset(split_dir, scores)
        if split == 'train':
            train, valid = split_train_valid_set(dataset)
        else:
            test = dataset


    files = ['train', 'valid', 'test']
    with open(f"{files[0]}.json", "w") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
        print(f"{files[0]}.json written with {len(train)} samples.")
    
    with open(f"{files[1]}.json", "w") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
        print(f"{files[1]}.json written with {len(valid)} samples.")
    
    with open(f"{files[2]}.json", "w") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
        print(f"{files[2]}.json written with {len(test)} samples.")



if __name__ == "__main__":
    main()
