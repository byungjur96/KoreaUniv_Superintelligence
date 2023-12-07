from datasets import Dataset, Image
import pandas as pd
import os

# ds = load_dataset('csv', data_files='/Dataset/Quilt-1M/train_dataset.csv')

csv_data = pd.read_csv('/Dataset/Quilt-1M/train_dataset.csv')
csv_file_path = '/Dataset/Quilt-1M/train_dataset.csv'
csv_data = pd.read_csv(csv_file_path)

img_dir = "/Dataset/Quilt-1M/train"
image_paths = csv_data['image_path']
new_image_paths = []
for image_path in image_paths:
    image_path = os.path.join(img_dir, image_path)
    new_image_paths.append(image_path)
captions = csv_data['caption']
# roi_text = csv_data['roi_text']
# noisy_text = csv_data['noisy_text']
# corrected_text = csv_data['corrected_text']

dataset = Dataset.from_dict({"image": new_image_paths, 
                             "text": captions,
                            #  "roi_text": roi_text,
                            #  "noisy_text": noisy_text,
                            #  "corrected_text": corrected_text
                             }).cast_column("image", Image())

# print(dataset)
