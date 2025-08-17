import json
import numpy as np
import re
from scipy.stats import pearsonr

# src: https://github.com/bicheng1225/HierTFR/blob/main/src/traintest_gop_hierTFR_v61_preFix_aspFix_pccLoss.py
# def valid_utt(audio_output, target):
#     mse = []
#     corr = []
#     for i in range(5):
#         cur_mse = np.mean(((audio_output[:, i] - target[:, i]) ** 2).numpy())
#         cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
#         mse.append(cur_mse)
#         corr.append(cur_corr)
#     return mse, corr



def compute_pcc(data):
    # Initialize dicts to hold scores
    pred_scores = {"accuracy": [], "completeness": [], "fluency": [], "prosodic": [], "total": []}
    true_scores = {"accuracy": [], "completeness": [], "fluency": [], "prosodic": [], "total": []}

    
    for item in data:
        pred = parse_scores(item["prediction"])
        true = parse_scores(item["answer"])
        for key in pred_scores:
            pred_scores[key].append(pred[key])
            true_scores[key].append(true[key])

    # Compute PCC
    pcc_results = {}
    for key in pred_scores.keys():
        if np.std(pred_scores[key]) == 0 or np.std(true_scores[key]) == 0:
            pcc_results[key] = None
        else:
            pcc_results[key] = pearsonr(pred_scores[key], true_scores[key])[0]

    return pcc_results


def parse_scores(score_str):
    # return {k.strip(): float(v.strip()) for k, v in (item.split(":") for item in score_str.split(","))}
    if not score_str or score_str.strip() == "":
        return {}
    
    try:
        return {
            k.strip(): float(v.strip()) 
            for k, v in (item.split(":") for item in score_str.split(","))
        }
    except:
        return {}
    
def word_to_number(word):
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'thirty': '30', 'thirty-two': '32',
        'thirty-four': '34', 'thirty-five': '35', 'thirty-six': '36',
        'thirty-seven': '37'
    }
    return number_words.get(word.lower(), word)
    
def extract_score(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return word_to_number(match.group(1))
    return None


def convert_rating_string(text):
    # 定義數字單詞到數字的映射
    patterns = {
        'accuracy': [
            r'accuracy[:,\s]+is\s+(\d+|[\w-]+)',
            r'accuracy[:,\s]+(\d+|[\w-]+)',
            r'accuracy[,\s]+(\d+|[\w-]+)'
        ],
        'completeness': [
            r'completeness[:,\s]+is\s+(\d+|[\w-]+)',
            r'completeness[:,\s]+(\d+|[\w-]+)',
            r'completeness[,\s]+(\d+|[\w-]+)'
        ],
        'fluency': [
            r'fluency[:,\s]+is\s+(\d+|[\w-]+)',
            r'fluency[:,\s]+(\d+|[\w-]+)',
            r'fluency[,\s]+(\d+|[\w-]+)'
        ],
        'prosody': [
            r'prosody[:,\s]+is\s+(\d+|[\w-]+)',
            r'prosody[:,\s]+(\d+|[\w-]+)',
            r'prosody[,\s]+(\d+|[\w-]+)'
        ],
        'total': [
            r'total[:,\s]+is\s+(\d+|[\w-]+)(?:\s+out of\s+\d+|[\w-]+)?',
            r'total[:,\s]+(\d+|[\w-]+)(?:\s+out of\s+\d+|[\w-]+)?',
            r'total[,\s]+(\d+|[\w-]+)(?:\s+out of\s+\d+|[\w-]+)?'
        ]
    }
    
    scores = {}
    for key in patterns:
        for pattern in patterns[key]:
            score = extract_score(text, pattern)
            print(score)
            if score is not None:
                scores[key] = score
                break
    
    if len(scores) == 5:
        return f"accuracy: {scores['accuracy']}, completeness: {scores['completeness']}, fluency: {scores['fluency']}, prosody: {scores['prosody']}, total: {scores['total']}"
    # else:
    #     print(text)
    return None


if __name__ == "__main__":
    file_path = 'predictions/8b-omni-10e-avg.json'

    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    invalid_count = 0
    for item in data:
        # form_str = convert_rating_string(text=item['prediction'])
        form_str = parse_scores(item['prediction'])
        if form_str == None:
            invalid_count += 1

    print(invalid_count)
    
    pcc_results = compute_pcc(data)
    
    print(pcc_results)