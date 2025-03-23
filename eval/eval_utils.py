import json
import os
import random
import torch
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

TOTAL_QUESTIONS_IN_DATASET = 34031  # Upper limit for question_id selection

def generate_random_question_ids(num_questions, output_file):
    """Generate random unique question IDs and save them to a file."""
    random_ids = random.sample(range(TOTAL_QUESTIONS_IN_DATASET), num_questions)
    with open(output_file, 'w') as f:
        json.dump(random_ids, f)
    return random_ids

def load_question_ids(question_id_file, num_questions):
    """Load question IDs from file or generate new ones if not provided."""
    if question_id_file:
        try:
            with open(question_id_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"[ERROR] Could not read {question_id_file}. Generating new question IDs.")
    
    output_file = "generated_question_ids.json"
    return generate_random_question_ids(num_questions, output_file)

def get_jsonl_file_paths(logs_dir="../logs"):
    """Find all output.jsonl files in ../logs/MODEL_NAME_log directories."""
    jsonl_files = []
    for model_name in os.listdir(logs_dir):
        model_log_dir = os.path.join(logs_dir, model_name)
        output_file = os.path.join(model_log_dir, "output.jsonl")
        if os.path.isfile(output_file):
            jsonl_files.append(output_file)
    return jsonl_files

def process_ncd(jsonl_file_paths, question_ids, item_to_knowledge_code):
    """Processes all student logs in each JSONL file (NCD format)."""
    all_users_data = []
    for user_id, jsonl_file_path in enumerate(jsonl_file_paths, start=1):
        logs = []
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                item_id = entry["question_id"]

                if item_id in question_ids:
                    logs.append({
                        "exer_id": len(logs) + 1,
                        "item_id": item_id,
                        "score": float(entry["correct"]),
                        "knowledge_code": item_to_knowledge_code.get(item_id, [])
                    })

        all_users_data.append({"user_id": user_id, "log_num": len(logs), "logs": logs})
    
    return all_users_data

def process_icdm(jsonl_file_paths, question_ids, num_students_per_file, sample_ratio):
    """Randomly samples logs to generate new log files (ICDM format)."""
    output_dir = "../ICDM_data"
    os.makedirs(output_dir, exist_ok=True)
    data_csv_path = os.path.join(output_dir, "data.csv")

    with open(data_csv_path, mode='w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(['user_id', 'item_id', 'score'])

        user_id = 0
        for jsonl_file_path in jsonl_file_paths:
            with open(jsonl_file_path, 'r') as file:
                lines = file.readlines()
                student_lines = [random.sample(lines, int(len(lines) * sample_ratio)) for _ in range(num_students_per_file)]

                for student_id, student_lines_subset in enumerate(student_lines):
                    for line in student_lines_subset:
                        entry = json.loads(line)
                        item_id = entry["question_id"]

                        if item_id in question_ids:
                            score = float(entry["correct"])
                            data_writer.writerow([user_id + student_id, item_id, score])
                
                user_id += num_students_per_file

    return data_csv_path

def process_few_shot(jsonl_file_paths, exercise_file_path, skill_mapping_path, record_file, info_file, num_samples=100):
    """Processes logs for Few-Shot learning, generating `recordings.jsonl` and `exercise_info.jsonl`."""
    os.makedirs(os.path.dirname(record_file), exist_ok=True)

    student_id = 1
    recordings = []

    for file_path in jsonl_file_paths:
        try:
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]

            sampled_data = random.sample(data, min(num_samples, len(data)))

            exercise_logs = [str(entry['question_id']) for entry in sampled_data]
            is_corrects = [str(entry['correct']) for entry in sampled_data]

            recordings.append({
                'student_id': student_id,
                'exercises_logs': exercise_logs,
                'is_corrects': is_corrects
            })
            student_id += 1

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {file_path} due to error: {e}")
            continue

    with open(record_file, 'w') as outfile:
        for record in recordings:
            outfile.write(json.dumps(record) + '\n')

    create_info_file(exercise_file_path, skill_mapping_path, info_file)

def create_info_file(exercise_file_path, skill_mapping_path, info_file):
    """Creates exercise information file for Few-Shot processing."""
    skill_mapping = {}
    try:
        with open(skill_mapping_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    skill_mapping[str(row[0])] = row[1]
    except FileNotFoundError:
        print(f"Error: {skill_mapping_path} not found.")
        return

    info_records = []
    try:
        with open(exercise_file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                exercise_id = str(entry['question_id'])
                exercise_desc = entry['stem']
                skill_ids = [str(skill_id) for skill_id in entry['ids']]
                skill_descs = [skill_mapping.get(skill_id, 'Unknown') for skill_id in skill_ids]
                skill_desc = ', '.join(skill_descs)

                info_records.append({
                    'exercise_id': exercise_id,
                    'exercise_desc': exercise_desc,
                    'skill_ids': skill_ids,
                    'skill_desc': skill_desc
                })

    except FileNotFoundError:
        print(f"Error: {exercise_file_path} not found.")
        return

    with open(info_file, 'w') as outfile:
        for record in info_records:
            outfile.write(json.dumps(record) + '\n')