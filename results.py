import os
import csv
import random
import matplotlib.pyplot as plt

result_dir = "./results"
experiments = ['zero_init', 'zero_init2', 'Batch1', 'Batch4', 'RadBERT', 'SapBERT']
description = ['Zero Init(Batch=1)', 'zero_init(Batch=4)', 'Only U-Net(Batch=1)', 'Only U-Net(Batch=4)', 'Text Encoder: RadBERT(Batch=4)', 'Text Encoder: SapBERT(Batch=4)']
DATASET_PATH = "/workspace/AAA740/Dataset"

def make_figure(file):
    plt.figure(figsize=(30, 7))
    
    plt.suptitle(text, fontsize=18)
    for i, experiment in enumerate(experiments):
        plt.subplot(1, len(experiments), i+1)
        plt.title(description[i])
        image = plt.imread(os.path.join(result_dir, experiment, file))
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(f'./figures2/{file}', bbox_inches='tight')
    plt.close()

with open(f'{DATASET_PATH}/test_dataset.csv', 'r') as file:
        csv_reader = csv.reader(file)
        global test_csv
        test_csv = []
        for row in csv_reader:
            test_csv.append(row)
            
print(f"Total {len(test_csv)} prompts ready.")

for experiment in experiments:
    print(f"[{experiment}] Total {len(os.listdir(os.path.join(result_dir, experiment)))} Images.")
    
if not os.path.exists("./figures2"):
    os.makedirs("./figures2")
    print(f"The folder './figures2' has been created.")
else:
    print(f"The folder './figures2' already exists.")

# num = random.randint(1, len(test_csv) - 1)
for num in range(1, len(test_csv) - 1):
    prompt_dic = test_csv[num]

    file_name = prompt_dic[2]
    text = prompt_dic[1]

    make_figure(file_name)
    if num % 10 == 0:
        print(num)