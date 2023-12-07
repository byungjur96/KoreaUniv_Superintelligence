import pandas as pd
import os

csv_data = pd.read_csv('/Dataset/Quilt-1M/train_dataset.csv')
print(len(csv_data))
exit()
csv_data = csv_data.drop_duplicates(subset=['image_path'])
print(len(csv_data))
exit()
csv_data.to_csv('/Dataset/Quilt-1M/new_train_dataset.csv', index=False)
