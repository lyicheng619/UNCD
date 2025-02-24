import os
import json
import random
import csv

# Input file paths
jsonl_file_paths = [
    'llama_exam/log.jsonl',
    'ga_300_exam/log.jsonl',
    'ga_600_exam/log.jsonl',
    'ga_900_exam/log.jsonl',
    'ga_1200_exam/log.jsonl',
    'ga_gdr_300_exam/log.jsonl',
    'ga_gdr_600_exam/log.jsonl',
    'ga_gdr_900_exam/log.jsonl',
    'ga_gdr_1200_exam/log.jsonl',
    'ga_klr_300_exam/log.jsonl',
    'ga_klr_600_exam/log.jsonl',
    'ga_klr_900_exam/log.jsonl',
    'ga_klr_1200_exam/log.jsonl',
    'npo_300_exam/log.jsonl',
    'npo_600_exam/log.jsonl',
    'npo_900_exam/log.jsonl',
    'npo_1200_exam/log.jsonl',
    'npo_gdr_300_exam/log.jsonl',
    'npo_gdr_600_exam/log.jsonl',
    'npo_gdr_900_exam/log.jsonl',
    'npo_gdr_1200_exam/log.jsonl',
    'npo_klr_300_exam/log.jsonl',
    'npo_klr_600_exam/log.jsonl',
    'npo_klr_900_exam/log.jsonl',
    'npo_klr_1200_exam/log.jsonl',
    'rmu_600_exam/log.jsonl',
    'rmu_1200_exam/log.jsonl',
    'rmu_1800_exam/log.jsonl',
    'rmu_2400_exam/log.jsonl',
    'tv_1500_exam/log.jsonl',
    'tv_3000_exam/log.jsonl',
    'tv_4500_exam/log.jsonl',
    'tv_6000_exam/log.jsonl',
]

# Path to the combined domain file
combined_domain_file_path = '../process/data_domain.jsonl'

# Output directory and file paths
output_dir = '../ncd_data'
qmatrix_file_path = os.path.join(output_dir, 'qmatrix.csv')
NUM_QUESTIONS = 5000  # Number of questions to select

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to create the Q-matrix and sample questions
def create_qmatrix_and_sample_questions(combined_domain_file_path, num_questions):
    """
    Samples questions and generates the Q-matrix.

    Args:
        combined_domain_file_path (str): Path to the combined domain file.
        num_questions (int): Number of questions to sample.

    Returns:
        list: A list of lists representing the Q-matrix.
        dict: A mapping from original item_id to new sequential IDs.
        list: Sampled questions as (new_id, original_line).
    """
    qmatrix = []
    sampled_item_ids = []
    item_id_mapping = {}  # Map original item_id -> new sequential id
    all_questions = []

    with open(combined_domain_file_path, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line)
                item_id = entry["item_id"]
                knowledge_codes = entry["domain_id"]  # List of knowledge codes

                # Generate binary vector for knowledge codes 1-14
                qmatrix_row = [1 if kc in knowledge_codes else 0 for kc in range(1, 15)]

                # Store the question and its Q-matrix row
                all_questions.append((item_id, qmatrix_row, line))
            except json.JSONDecodeError:
                print("Skipping invalid JSON in combined_domain.jsonl")
                continue

    # Randomly sample `num_questions` questions
    sampled_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
    for new_id, (original_item_id, qmatrix_row, original_line) in enumerate(sampled_questions):
        item_id_mapping[original_item_id] = new_id
        sampled_item_ids.append((new_id, original_line))
        qmatrix.append(qmatrix_row)

    return qmatrix, item_id_mapping, sampled_item_ids

# Function to generate s1.jsonl files for each folder
def generate_s1_jsonl_files(jsonl_file_paths, item_id_mapping):
    """
    Generates `s1.jsonl` files for each folder in jsonl_file_paths.

    Args:
        jsonl_file_paths (list): List of paths to student log files.
        item_id_mapping (dict): Mapping of original item_id to new sequential IDs.
    """
    for jsonl_file_path in jsonl_file_paths:
        # Extract the folder path
        folder_path = os.path.dirname(jsonl_file_path)
        s1_file_path = os.path.join(folder_path, 's1.jsonl')

        # Generate the s1.jsonl file
        with open(jsonl_file_path, 'r') as input_file, open(s1_file_path, 'w') as output_file:
            for line in input_file:
                try:
                    entry = json.loads(line)
                    original_item_id = entry["item_id"]

                    # Write only questions with a new ID in `item_id_mapping`
                    if original_item_id in item_id_mapping:
                        # Update the item_id to the new sequential ID
                        entry["item_id"] = item_id_mapping[original_item_id]
                        output_file.write(json.dumps(entry) + "\n")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {jsonl_file_path}")
                    continue

        print(f"s1.jsonl has been generated at {s1_file_path}")

# Create the Q-matrix and sample questions
qmatrix, item_id_mapping, sampled_item_ids = create_qmatrix_and_sample_questions(
    combined_domain_file_path, NUM_QUESTIONS
)

# Generate s1.jsonl files for each folder
generate_s1_jsonl_files(jsonl_file_paths, item_id_mapping)

# Write the Q-matrix to a CSV file
with open(qmatrix_file_path, 'w', newline='') as qmatrix_file:
    writer = csv.writer(qmatrix_file)
    writer.writerows(qmatrix)

# Print summary to the console
print(f"Q-matrix has been saved to {qmatrix_file_path}")
print(f"s1.jsonl files have been generated for all folders.")