import json
import random
import csv 
# List of JSONL file paths
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
    'rmu_600_exam/output.jsonl',
    'rmu_1200_exam/output.jsonl',
    'rmu_1800_exam/output.jsonl',
    'rmu_2400_exam/output.jsonl',
    'tv_1500_exam/output.jsonl',
    'tv_3000_exam/output.jsonl',
    'tv_4500_exam/output.jsonl',
    'tv_6000_exam/output.jsonl',
]
exercise_file_path='../process/combined_data.jsonl'
skill_mapping_path='../process/skill_mapping.csv'

# Define the output file paths
record_file = '../few-shot/datasets/moderate/fs_data/recordings.jsonl'
info_file='../few-shot/datasets/moderate/fs_data/moderate_exercise_info.jsonl'
# Number of sampled exercises per file
NUM = 100  # Example, adjust based on your needs


def add_item_id_to_jsonl_file(file_path):
    # Read the file lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Open the file again for writing
    with open(file_path, 'w', encoding='utf-8') as file:
        for line_number, line in enumerate(lines):
            try:
                # Parse each line as JSON
                data = json.loads(line)
                
                # Add 'item_id' if it doesn't exist
                if 'item_id' not in data:
                    data['item_id'] = line_number
                
                # Write the modified JSON back to the file
                file.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                # Skip lines with JSON errors
                continue


# Function to process each JSONL file and sample exercise logs
def process_jsonl_files(jsonl_file_paths, record_file, num_samples):
    student_id = 1  # Start student_id from 1
    recordings = []

    for file_path in jsonl_file_paths:
        try:
            with open(file_path, 'r') as f:
                # Read all lines from the current JSONL file
                data = [json.loads(line) for line in f]

            # If the file has fewer entries than NUM, sample all
            sampled_data = random.sample(data, min(num_samples, len(data)))

            # Create a new record for the sampled data
            exercise_logs = [str(entry['item_id']) for entry in sampled_data]
            is_corrects = [str(entry['correct']) for entry in sampled_data]

            # Append the new student record
            recordings.append({
                'student_id': student_id,
                'exercises_logs': exercise_logs,
                'is_corrects': is_corrects
            })

            student_id += 1  # Increment the student_id for the next file

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # If there's any error, log and skip this file
            print(f"Skipping {file_path} due to error: {e}")
            continue

    # Write the recordings to the output JSONL file
    with open(record_file, 'w') as outfile:
        for record in recordings:
            outfile.write(json.dumps(record) + '\n')


def create_info_file(exercise_file_path, skill_mapping_path, info_file):
    # Load the skill mapping from the CSV file (no headers)
    skill_mapping = {}
    try:
        with open(skill_mapping_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    # First column is skill_id (int), second is skill_name (str)
                    skill_id = str(row[0])  # Convert skill_id to string for consistency
                    skill_name = row[1]     # The skill name/description
                    skill_mapping[skill_id] = skill_name
                else:
                    print(f"Skipping malformed row in skill mapping: {row}")
    except FileNotFoundError as e:
        print(f"Error: {skill_mapping_path} not found.")
        return
    except Exception as e:
        print(f"Error reading skill mapping CSV: {e}")
        return

    # Process the exercise file (JSONL format)
    info_records = []
    try:
        with open(exercise_file_path, 'r') as f:
            for line in f:
                try:
                    # Parse the JSON object from the line
                    entry = json.loads(line)
                    
                    # Extract the required fields
                    exercise_id = str(entry['item_id'])  # Convert item_id to string
                    exercise_desc = entry['stem']       # Use the 'stem' field as the description
                    skill_ids = [str(skill_id) for skill_id in entry['ids']]  # Convert ids to strings
                    
                    # Map skill IDs to skill descriptions
                    skill_descs = [skill_mapping.get(skill_id, 'Unknown') for skill_id in skill_ids]

                    # Combine all skill descriptions into a single string
                    skill_desc = ', '.join(skill_descs)

                    # Create the new record
                    info_records.append({
                        'exercise_id': exercise_id,
                        'exercise_desc': exercise_desc,
                        'skill_ids': skill_ids,
                        'skill_desc': skill_desc
                    })

                except KeyError as e:
                    print(f"Missing key in entry: {e}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue
    except FileNotFoundError as e:
        print(f"Error: {exercise_file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading exercise file: {e}")
        return

    # Write the info records to the output JSONL file
    try:
        with open(info_file, 'w') as outfile:
            for record in info_records:
                outfile.write(json.dumps(record) + '\n')
        print(f"Info file written to {info_file}")
    except Exception as e:
        print(f"Error writing to info file: {e}")

add_item_id_to_jsonl_file(exercise_file_path)
process_jsonl_files(jsonl_file_paths, record_file, NUM)
create_info_file(exercise_file_path, skill_mapping_path, info_file)
