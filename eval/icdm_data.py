import json
import csv
import random

jsonl_file_paths=[
    'llama_exam/s1.jsonl',
    'ga_300_exam/s1.jsonl',
    'ga_600_exam/s1.jsonl',
    'ga_900_exam/s1.jsonl',
    'ga_1200_exam/s1.jsonl',
    'ga_gdr_300_exam/s1.jsonl',
    'ga_gdr_600_exam/s1.jsonl',
    'ga_gdr_900_exam/s1.jsonl',
    'ga_gdr_1200_exam/s1.jsonl',
    'ga_klr_300_exam/s1.jsonl',
    'ga_klr_600_exam/s1.jsonl',
    'ga_klr_900_exam/s1.jsonl',
    'ga_klr_1200_exam/s1.jsonl',
    'npo_300_exam/s1.jsonl',
    'npo_600_exam/s1.jsonl',
    'npo_900_exam/s1.jsonl',
    'npo_1200_exam/s1.jsonl',
    'npo_gdr_300_exam/s1.jsonl',
    'npo_gdr_600_exam/s1.jsonl',
    'npo_gdr_900_exam/s1.jsonl',
    'npo_gdr_1200_exam/s1.jsonl',
    'npo_klr_300_exam/s1.jsonl',
    'npo_klr_600_exam/s1.jsonl',
    'npo_klr_900_exam/s1.jsonl',
    'npo_klr_1200_exam/s1.jsonl',
    'rmu_600_exam/s1.jsonl',
    'rmu_1200_exam/s1.jsonl',
    'rmu_1800_exam/s1.jsonl',
    'rmu_2400_exam/s1.jsonl',
    'tv_1500_exam/s1.jsonl',
    'tv_3000_exam/s1.jsonl',
    'tv_4500_exam/s1.jsonl',
    'tv_6000_exam/s1.jsonl',
]


# new_list=[
#     'llama_exam/s1.jsonl',
#     'ga_1200_exam/s1.jsonl',
#     'ga_gdr_1200_exam/s1.jsonl',
#     'ga_klr_1200_exam/s1.jsonl',
#     'npo_1200_exam/s1.jsonl',
#     'npo_gdr_1200_exam/s1.jsonl',
#     'npo_klr_1200_exam/s1.jsonl',
#     'rmu_2400_exam/s1.jsonl',
#     'tv_6000_exam/s1.jsonl',
# ]


new_list=[
    'llama_exam/s1.jsonl',
    'ga_300_exam/s1.jsonl',
    'ga_600_exam/s1.jsonl',
    'ga_900_exam/s1.jsonl',
    'ga_1200_exam/s1.jsonl',
    'ga_gdr_300_exam/s1.jsonl',
    'ga_gdr_600_exam/s1.jsonl',
    'ga_gdr_900_exam/s1.jsonl',
    'ga_gdr_1200_exam/s1.jsonl',
    'ga_klr_300_exam/s1.jsonl',
    'ga_klr_600_exam/s1.jsonl',
    'ga_klr_900_exam/s1.jsonl',
    'ga_klr_1200_exam/s1.jsonl',
    'npo_300_exam/s1.jsonl',
    'npo_600_exam/s1.jsonl',
    'npo_900_exam/s1.jsonl',
    'npo_1200_exam/s1.jsonl',
    'npo_gdr_300_exam/s1.jsonl',
    'npo_gdr_600_exam/s1.jsonl',
    'npo_gdr_900_exam/s1.jsonl',
    'npo_gdr_1200_exam/s1.jsonl',
    'npo_klr_300_exam/s1.jsonl',
    'npo_klr_600_exam/s1.jsonl',
    'npo_klr_900_exam/s1.jsonl',
    'npo_klr_1200_exam/s1.jsonl',
    'rmu_600_exam/s1.jsonl',
    'rmu_1200_exam/s1.jsonl',
    'rmu_1800_exam/s1.jsonl',
    'rmu_2400_exam/s1.jsonl',
    'tv_1500_exam/s1.jsonl',
    'tv_3000_exam/s1.jsonl',
    'tv_4500_exam/s1.jsonl',
    'tv_6000_exam/s1.jsonl',
]


s1_item_csv_file_path = '../icdm_data/item.csv'
s1_data_csv_file_path = '../icdm_data/data.csv'
mapping_s1_file_path = '../icdm_data/mapping.txt'

def add_item_id_to_jsonl_files(file_paths):
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        with open(file_path, 'w', encoding='utf-8') as file:
            for line_number, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    if 'item_id' not in data:
                        data['item_id'] = line_number
                    file.write(json.dumps(data) + '\n')
                except json.JSONDecodeError:
                    # Skip lines with JSON errors
                    continue

def process_jsonl_files(num_students_per_file, sample_ratio, new_list_sample_ratio):
    user_id = 0
    new_list_user_ids = []
    index_mapping = {}

    with open(s1_item_csv_file_path, mode='w', newline='') as item_file, \
         open(s1_data_csv_file_path, mode='w', newline='') as data_file:
        item_writer = csv.writer(item_file)
        data_writer = csv.writer(data_file)
        
        # Write headers
        item_writer.writerow(['user_id', 'item_id', 'knowledge_code'])
        data_writer.writerow(['user_id', 'item_id', 'score'])

        # Process all files
        for jsonl_file_path in jsonl_file_paths:
            with open(jsonl_file_path, 'r') as file:
                lines = file.readlines()
                # Divide into specified number of students, each with the specified sample ratio
                student_lines = [random.sample(lines, int(len(lines) * sample_ratio)) for _ in range(num_students_per_file)]

                for student_id, student_lines_subset in enumerate(student_lines):
                    for line in student_lines_subset:
                        process_line(line, data_writer, item_writer, user_id + student_id)
                
                user_id += num_students_per_file  # Increment user_id for the next set of students

        # Process new_list files as additional single students
        for index, jsonl_file_path in enumerate(new_list):
            with open(jsonl_file_path, 'r') as file:
                lines = file.readlines()
                sampled_lines = random.sample(lines, int(len(lines) * new_list_sample_ratio))

                current_user_id = user_id
                new_list_user_ids.append(current_user_id)
                index_mapping[current_user_id] = index  # Map user_id to index of new_list
                for line in sampled_lines:
                    process_line(line, data_writer, item_writer, current_user_id)
            
            user_id += 1  # Increment user_id for each additional student

    return new_list_user_ids, user_id, index_mapping  # Return the mapping as well

def process_line(line, data_writer, item_writer, user_id):
    try:
        entry = json.loads(line)
        knowledge_codes = entry["ids"]
        item_id = entry["item_id"]

        # Check if any knowledge code exceeds 100
        if any(kc > 100 for kc in knowledge_codes):
            return  # Skip this line if the condition is met
        
        score = float(entry["correct"])  # Convert the correct flag into float score
        
        # Write to data.csv
        data_writer.writerow([user_id, item_id, score])
        
        # Format knowledge codes into a string resembling a Python list
        formatted_knowledge_codes = f"[{','.join(map(str, knowledge_codes))}]"
        # Add user_id when writing to item.csv
        item_writer.writerow([user_id, item_id, formatted_knowledge_codes])
            
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON")

def mix_and_shuffle_user_ids(new_list_user_ids, total_user_ids, index_mapping):
    total_students = total_user_ids
    all_user_ids = list(range(total_students))

    # Shuffle all user IDs
    #random.shuffle(all_user_ids)

    # Map new_list user IDs to their shuffled positions
    mapping = {original: all_user_ids[original] for original in new_list_user_ids}
    new_ids_after_switch = [all_user_ids[original] for original in new_list_user_ids]

    # Map new IDs to their original new_list indices
    new_index_mapping = {new_id: index_mapping[original] for original, new_id in mapping.items()}

    # Save the mapping and new IDs to a text file
    with open(mapping_s1_file_path, 'w') as mapping_file:
        mapping_file.write("Original -> Shuffled\n")
        for original, shuffled in mapping.items():
            mapping_file.write(f"{original} -> {shuffled}\n")
        mapping_file.write("\nNew IDs after switching:\n")
        mapping_file.write(", ".join(map(str, new_ids_after_switch)))
        mapping_file.write(f"\nTotal number of IDs generated: {total_user_ids}\n")
        mapping_file.write("\nNew ID to new_list index mapping:\n")
        for new_id, index in new_index_mapping.items():
            mapping_file.write(f"new_id {new_id} -> new_list index {index}\n")

    print("New list ID mapping and new IDs saved to mapping.txt")

def check_max_item_id(data_csv_path):
    max_item_id = -1

    with open(data_csv_path, mode='r', encoding='utf-8') as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            item_id = int(row['item_id'])
            if item_id > max_item_id:
                max_item_id = item_id

    print(f"The maximum item_id in {data_csv_path} is: {max_item_id}")


add_item_id_to_jsonl_files(jsonl_file_paths)
# num_students_per_file =10
# sample_ratio = 0.5     good results

# num_students_per_file =50
# sample_ratio = 0.5
# new_list_sample_ratio = 0.5   和上面差不多


# num_students_per_file =15
# sample_ratio = 0.5
# new_list_sample_ratio = 0.1

num_students_per_file =15
sample_ratio = 0.5
new_list_sample_ratio = 0.1


new_list_user_ids, total_user_ids, index_mapping = process_jsonl_files(num_students_per_file, sample_ratio, new_list_sample_ratio)
print("CSV files have been created successfully.")
mix_and_shuffle_user_ids(new_list_user_ids, total_user_ids, index_mapping)


check_max_item_id(s1_data_csv_file_path)