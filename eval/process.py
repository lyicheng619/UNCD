import json
import os

# List of file paths
jsonl_file_paths = [
    'llama_exam/output.jsonl',
    'ga_300_exam/output.jsonl',
    'ga_600_exam/output.jsonl',
    'ga_900_exam/output.jsonl',
    'ga_1200_exam/output.jsonl',
    'ga_gdr_300_exam/output.jsonl',
    'ga_gdr_600_exam/output.jsonl',
    'ga_gdr_900_exam/output.jsonl',
    'ga_gdr_1200_exam/output.jsonl',
    'ga_klr_300_exam/output.jsonl',
    'ga_klr_600_exam/output.jsonl',
    'ga_klr_900_exam/output.jsonl',
    'ga_klr_1200_exam/output.jsonl',
    'ga_klr_new_exam/output.jsonl',
    'npo_300_exam/output.jsonl',
    'npo_600_exam/output.jsonl',
    'npo_900_exam/output.jsonl',
    'npo_1200_exam/output.jsonl',
    'npo_gdr_300_exam/output.jsonl',
    'npo_gdr_600_exam/output.jsonl',
    'npo_gdr_900_exam/output.jsonl',
    'npo_gdr_1200_exam/output.jsonl',
    'npo_klr_300_exam/output.jsonl',
    'npo_klr_600_exam/output.jsonl',
    'npo_klr_900_exam/output.jsonl',
    'npo_klr_1200_exam/output.jsonl',
    'npo_klr_new_exam/output.jsonl',
    'rmu_600_exam/output.jsonl',
    'rmu_1200_exam/output.jsonl',
    'rmu_1800_exam/output.jsonl',
    'rmu_2400_exam/output.jsonl',
    'rmu_new_exam/output.jsonl',
    'rmu_tv_exam/output.jsonl',
    'tv_1500_exam/output.jsonl',
    'tv_3000_exam/output.jsonl',
    'tv_4500_exam/output.jsonl',
    'tv_6000_exam/output.jsonl',
    'tv_new_exam/output.jsonl'
    
]

# Path to the example file
example_file = "../process/data_domain.jsonl"

# Read the example file into a dictionary keyed by item_id
example_data = {}
with open(example_file, 'r') as ef:
    for line in ef:
        example_line = json.loads(line)
        item_id = example_line['item_id']
        domain_id = example_line.get('domain_id')
        difficulty = example_line.get('difficulty')
        example_data[item_id] = {
            'domain_id': domain_id,
            'difficulty': difficulty
        }

# Cleanup previously generated files
for file_path in jsonl_file_paths:
    folder, original_file_name = os.path.split(file_path)
    new_file_path = os.path.join(folder, 'log.jsonl')  # Name of the new file

    # If the file exists, remove it
    if os.path.exists(new_file_path):
        os.remove(new_file_path)
        print(f"Removed previously generated file: {new_file_path}")

# Process each file in the list
for file_path in jsonl_file_paths:
    folder, original_file_name = os.path.split(file_path)
    new_file_path = os.path.join(folder, 'log.jsonl')  # Name of the new file

    # Read the original file and add the `domain_id` and `difficulty` fields
    with open(file_path, 'r') as infile, open(new_file_path, 'w') as outfile:
        for line in infile:
            original_line = json.loads(line)
            item_id = original_line['item_id']

            # If the item_id is in the example data, copy `domain_id` and `difficulty`
            if item_id in example_data:
                original_line.update(example_data[item_id])

            # Write the updated line to the new file
            outfile.write(json.dumps(original_line) + '\n')

print("Processing complete. New files with example data have been created.")