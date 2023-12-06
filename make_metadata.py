import csv
import json

# Specify the path to your CSV file
csv_file_path = '../Dataset/train_dataset.csv'

# Open the CSV file
with open(csv_file_path, 'r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)

    # Initialize a list to store the data
    csv_data = []

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Append the row to the list
        csv_data.append(row)

# Now csv_data contains the CSV information
print(csv_data[:2])

metadata = []

print(f"CSV data with {len(csv_data)} lines")

for d in csv_data:
    if d[0] == 'Unnamed: 0':
        continue
    # print(d)
    sample = {
        "file_name" : d[2],
        "text" : d[1]
    }
    metadata.append(sample)
    
print(f"Metadata saved in {len(metadata)} lines")

jsonl_file_path = '../Dataset/train/metadata.jsonl'

with open(jsonl_file_path, 'w') as file:
    for record in metadata:
        file.write(json.dumps(record) + '\n')
        
read_data = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        record = json.loads(line)
        read_data.append(record)

print(read_data[:5])