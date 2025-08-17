import json
import os


for split in ["train", "valid", "test"]:

    with open(f"{split}.json", "r") as f1:
        s1_data = json.load(f1)

    with open(f"speech_units/km_labels/{split}_0_1.km", "r") as f2:
        unit_strings = [line.strip() for line in f2 if line.strip()]

    questions, answers = [], []
    new_speech_path = "speechocean/WAVE/input_instruct.wav"
    
    for i, item in enumerate(s1_data):
        human = item['conversations'][0]['value']
        scores = item['conversations'][1]['value']
        transcript = human.split('\n')[2]

        questions_item = {
            "id": item["id"],
            "speech": new_speech_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<speech>\nPlease say the following script based on the target pronunciation scores. Each score ranges from 0 to 10 and includes the following aspects: accuracy, completeness, fluency, prosody, and total.\nScript: {transcript}\nTarget scores: {scores}"
                },
                {
                    "from": "assistant",
                    "value": transcript
                }
            ]
        }
        answers_item = {
            "question_id": item["id"], 
            "prediction": transcript,
            "prediction_units": unit_strings[i],
            "answer": transcript
        }
        questions.append(questions_item)
        answers.append(answers_item)

    with open(f"stage2_data/{split}_questions.json", "w", encoding="utf-8") as fq:
        json.dump(questions, fq, indent=2, ensure_ascii=False)

    with open(f"stage2_data/{split}_answers.json", "w", encoding="utf-8") as fa:
        for item in answers:
            fa.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(questions)} entries")
