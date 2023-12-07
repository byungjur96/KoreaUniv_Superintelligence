from datasets import load_dataset
from PIL import Image
import pandas as pd
import os

# dataset = load_dataset(
#     'csv', data_files='/Dataset/Quilt-1M/quilt_1M_lookup.csv', split='train'
# )

# image_dataset = []
# img_dir = '/Dataset/Quilt-1M/train'
# for image_file in os.listdir(img_dir):
#     image_path = os.path.join(img_dir, image_file)
#     img = Image.open(image_path)
#     # You can apply any necessary preprocessing to the image here
#     image_dataset.append(img)

csv_data = pd.read_csv('/Dataset/Quilt-1M/quilt_1M_lookup.csv')
csv_data = csv_data.drop_duplicates(subset=['image_path'])

# train, test
image_folder = '/Dataset/Quilt-1M/test'
image_files = os.listdir(image_folder)

valid_image_files = []
for image_name in csv_data['image_path']:
    # path_image_name = os.path.join(image_folder, image_name)
    # print(path_image_name)
    # exit()
    if (image_name in image_files) and (image_name not in valid_image_files):
        valid_image_files.append(image_name)

print(len(valid_image_files))
# exit()

combined_dataset = pd.merge(csv_data, pd.DataFrame(valid_image_files, columns=['image_path']), on='image_path', how='inner')
# csv_data_no_duplicates = csv_data.drop_duplicates(subset=['image_path'])
# combined_dataset = pd.merge(csv_data_no_duplicates, pd.DataFrame(valid_image_files, columns=['image_path']), on='image_path', how='inner')
combined_dataset.to_csv('/Dataset/Quilt-1M/test_dataset.csv', index=False)
