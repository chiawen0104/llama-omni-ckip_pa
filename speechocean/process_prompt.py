import json
import random
import copy

def generate_random_pronunciation_scores():
    accuracy = random.randint(1, 10)
    completeness = random.randint(1, 10)
    fluency = random.randint(1, 10)
    prosodic = random.randint(1, 10)
    total = random.randint(1, 10)
    target_str = f"accuracy: {accuracy}, completeness: {completeness}, fluency: {fluency}, prosodic: {prosodic}, total: {total}"
    
    return target_str



if __name__ == "__main__":
    file_path = 'test.json'

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    for item in data:
        human = item['conversations'][0]['value']
        assistant = item['conversations'][1]['value']
        transcript = human.split('\n')[2]
        # for accuracy in range(11):
        #     for fluency in range(11):
        #         completeness = 10
        #         prosodic = 7
        #         total = 7
        #         target_str = f"accuracy: {accuracy}, completeness: {completeness}, fluency: {fluency}, prosodic: {prosodic}, total: {total}"
        prompt = f"<speech>\nListen to the input speech, please assess the speaker's pronunciation of the following script:\n{transcript}\nScore the accuracy, completeness, fluency, prosody, and total at the sentence level. Each score should range from 0 to 10."
        # prompt = "You are given one speech utterance at a time. Transcribe the speech to text verbatim.\nOutput Example 1: This is a book.\nOutput Example 2: Israel’s military says that Iran has launched more than 100 drones toward Israeli territory in what is expected to be the first stage of a much larger counter-attack."
        # prompt = f"<speech> Listen to this audio and compare it with the transcript:\n{transcript}\nProvide feedback on: 1. Which words were mispronounced 2. Your analysis of the speaker's pronunciation"
                # prompt = f"<speech>\nPlease listen to the input speech and refer to its pronunciation scores (reference scores), and generate new speech that reads the same script but matches the target scores.\nScript: {transcript}\nReference Scores: {assistant}\nTarget scores : {target_str}"
                # new_item = copy.deepcopy(item)
                # new_item['conversations'][0]['value'] = prompt
                # new_item['conversations'][1]['value'] = assistant
        item['conversations'][0]['value'] = prompt
        item['conversations'][1]['value'] = assistant
                # output.append(new_item)

    # print(len(output))
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)