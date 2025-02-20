
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

# Path to the data_domain.jsonl file
data_domain_file_path = '../process/data_domain.jsonl'

# Output directory and file paths
output_dir = '../ncd_data'
os.makedirs(output_dir, exist_ok=True)
total_data_path = os.path.join(output_dir, 'total.json')
total_qmatrix_path = os.path.join(output_dir, 'total_qmatrix.csv')
output_json_file_path = os.path.join(output_dir, 'ncd_data.json')
config_file_path = os.path.join(output_dir, 'config.txt')
qmatrix_file_path = os.path.join(output_dir, 'qmatrix.csv')

NUM_QUESTIONS = 5000  # Number of questions to select

# Function to create the Q-matrix and map item_ids to knowledge codes
def create_qmatrix_and_map_knowledge_codes(data_domain_file_path, num_questions=None):
    """
    Creates the Q-matrix for the selected questions or all questions if num_questions is None.
    Also creates a mapping of item_id to its knowledge codes.

    Args:
        data_domain_file_path (str): Path to the data_domain.jsonl file.
        num_questions (int, optional): Number of questions to select. If None, include all questions.

    Returns:
        list: A list of lists representing the Q-matrix.
        set: The set of selected question IDs.
        dict: A dictionary mapping item_id to its knowledge codes.
    """
    qmatrix = []
    selected_item_ids = set()
    item_to_knowledge_code = {}
    
    with open(data_domain_file_path, 'r') as file:
        all_questions = []
        for line in file:
            try:
                entry = json.loads(line)
                item_id = entry["item_id"]
                knowledge_codes = entry["domain_id"]  # List of knowledge codes

                # Generate binary vector for knowledge codes 1-14
                qmatrix_row = [1 if kc in knowledge_codes else 0 for kc in range(1, 15)]

                # Store the mapping of item_id to the knowledge codes
                item_to_knowledge_code[item_id] = knowledge_codes

                all_questions.append((item_id, qmatrix_row))
            except json.JSONDecodeError:
                print("Skipping invalid JSON in data_domain.jsonl")
                continue
        
        # If num_questions is specified, randomly select questions
        if num_questions:
            selected_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
        else:
            selected_questions = all_questions  # Include all questions
        
        for item_id, qmatrix_row in selected_questions:
            selected_item_ids.add(item_id)
            qmatrix.append(qmatrix_row)
    
    return qmatrix, selected_item_ids, item_to_knowledge_code

# Function to process each student's log file
def process_student_logs(jsonl_file_paths, selected_item_ids=None, item_to_knowledge_code=None):
    """
    Processes the student logs to filter responses to selected questions or include all logs.

    Args:
        jsonl_file_paths (list): List of paths to student log files.
        selected_item_ids (set, optional): Set of selected question IDs. If None, include all logs.
        item_to_knowledge_code (dict, optional): Mapping of item_id to its knowledge codes.

    Returns:
        list: A list of dictionaries containing student logs.
    """
    all_users_data = []
    for user_id, jsonl_file_path in enumerate(jsonl_file_paths, start=1):
        logs = []
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    item_id = entry["item_id"]
                    
                    # Include responses based on selected_item_ids or include all
                    if selected_item_ids is None or item_id in selected_item_ids:
                        log_entry = {
                            "exer_id": len(logs) + 1,  # Assign a sequential exercise ID
                            "item_id": item_id,
                            "score": float(entry["correct"]),
                        }
                        if item_to_knowledge_code:
                            log_entry["knowledge_code"] = item_to_knowledge_code[item_id]
                        logs.append(log_entry)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {jsonl_file_path}")
                    continue
        
        # Add user's data to the overall list
        all_users_data.append({
            "user_id": user_id,
            "log_num": len(logs),
            "logs": logs
        })
    
    return all_users_data

# Create the Q-matrix and map item_ids to knowledge codes for selected questions
qmatrix, selected_item_ids, item_to_knowledge_code = create_qmatrix_and_map_knowledge_codes(
    data_domain_file_path, NUM_QUESTIONS
)

# Create the Q-matrix and map item_ids to knowledge codes for all questions
total_qmatrix, _, total_item_to_knowledge_code = create_qmatrix_and_map_knowledge_codes(
    data_domain_file_path, None
)

# Process the student logs for selected questions
all_users_data = process_student_logs(jsonl_file_paths, selected_item_ids, item_to_knowledge_code)

# Process the student logs for all questions
total_users_data = process_student_logs(jsonl_file_paths)

# Write the Q-matrix (selected questions) to a CSV file
with open(qmatrix_file_path, 'w', newline='') as qmatrix_file:
    writer = csv.writer(qmatrix_file)
    writer.writerows(qmatrix)

# Write the total Q-matrix (all questions) to a CSV file
with open(total_qmatrix_path, 'w', newline='') as total_qmatrix_file:
    writer = csv.writer(total_qmatrix_file)
    writer.writerows(total_qmatrix)

# Write the filtered student logs (selected questions) to a JSON file
with open(output_json_file_path, 'w') as json_file:
    json.dump(all_users_data, json_file, indent=4)

# Write the total student logs (all questions) to a JSON file
with open(total_data_path, 'w') as json_file:
    json.dump(total_users_data, json_file, indent=4)

# Write the configuration summary to a text file
with open(config_file_path, 'w') as config_file:
    config_file.write(f"Number of students: {len(jsonl_file_paths)}\n")
    config_file.write(f"Number of selected questions: {NUM_QUESTIONS}\n")
    config_file.write(f"Number of total questions: {len(total_qmatrix)}\n")
    config_file.write(f"Number of unique knowledge concepts: 14\n")

# Print summary to the console
print(f"Q-matrix (selected) has been saved to {qmatrix_file_path}")
print(f"Q-matrix (total) has been saved to {total_qmatrix_path}")
print(f"Filtered student logs have been saved to {output_json_file_path}")
