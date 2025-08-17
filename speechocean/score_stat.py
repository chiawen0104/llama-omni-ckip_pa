import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

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
    

if __name__ == "__main__":

    subsets = ["train", "valid", "test"]

    for s in subsets:
        js_path = f"{s}.json" 

        # 讀入原始資料
        with open(js_path, "r", encoding="utf-8") as f:
            data = json.load(f)


        scores = []

        for item in data:
            score_str = item["conversations"][1]["value"]
            score_dict = parse_scores(score_str)
            scores.append(score_dict)

        score_keys = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']

        stats = {}
        for key in score_keys:
            values = [item[key] for item in scores]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        print(s)
        print(stats)

        # # 為每一個分數類型計算分布
        # score_distributions = {key: Counter() for key in score_keys}
        # for entry in scores:
        #     for key in score_keys:
        #         score_distributions[key][entry[key]] += 1

        # # 繪製每個分數的長條圖
        # fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
        # for ax, key in zip(axes, score_keys):
        #     dist = score_distributions[key]
        #     xs = sorted(dist.keys())
        #     ys = [dist[x] for x in xs]
        #     ax.bar(xs, ys)
        #     ax.set_title(key)
        #     ax.set_xticks(range(0, 11))
        #     ax.set_ylim(0, max(ys) + 1)

        # fig.suptitle(f'{s} Score Distributions')
        # plt.tight_layout()
        # fname = f"images/{s}_score.png"
        # plt.savefig(fname)