import os
import csv
import torch
import matplotlib.pyplot as plt
from torchmetrics.multimodal.clip_score import CLIPScore


CSV_PATH = "/Dataset/Quilt-1M"
DATASET_PATH = "/workspace/results_image/results/Vanila"
DATASET = DATASET_PATH.split("results/")[-1]

# Batch1 / Batch4 / RadBERT
# SapBERT / zero_init / zero_init2

# Load CSV file of test dataset
with open(f'{CSV_PATH}/test_dataset.csv', 'r') as file:
    csv_reader = csv.reader(file)
    test_csv = []
    for row in csv_reader:
        test_csv.append(row)

img_list = os.listdir(DATASET_PATH)
scores = torch.zeros(len(img_list))

for i, prompt_num in enumerate(range(1, len(test_csv))):
    prompt_line = test_csv[prompt_num]

    prompt_id = prompt_line[0]
    prompt_text = prompt_line[1]
    prompt_file = prompt_line[2]

    print(f"[{prompt_num}/{len(test_csv)}] {prompt_text} ({prompt_id})")

    ori = plt.imread(f"{DATASET_PATH}/{prompt_file}")
    ori = torch.from_numpy(ori) #.to("cuda")

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score = metric(ori, prompt_text[:77])
    score.detach()
    scores[i] = score

CLIP_SCORE = torch.mean(scores)
print(f"CLIP score of {DATASET}: {CLIP_SCORE:.3f}")
