import argparse
import json
from eval_utils import (
    load_question_ids, get_jsonl_file_paths, create_qmatrix, process_ncd, process_icdm, process_few_shot
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_list_file", type=str, help="Path to the text file containing model paths.")
    parser.add_argument("devices", type=str, help="Comma-separated CUDA device numbers (e.g., '0,1,2,3').")
    parser.add_argument("--method", type=str, choices=["NCD", "ICDM", "few-shot"], required=True, help="Method for data processing.")
    parser.add_argument("--num_questions", type=int, default=1000, help="Number of random questions to process.")
    parser.add_argument("--num_students_per_file", type=int, default=15)
    parser.add_argument("--sample_ratio", type=float, default=0.5)

    args = parser.parse_args()

    if args.method == "few-shot":
        args.num_questions = 100  # Override num_questions for few-shot

    question_ids = set(load_question_ids(None, args.num_questions))
    jsonl_file_paths = get_jsonl_file_paths()

    if args.method == "NCD":
        qmatrix, _, item_to_knowledge_code = create_qmatrix("../process/data_domain.jsonl", args.num_questions)
        data = process_ncd(jsonl_file_paths, question_ids, item_to_knowledge_code)
        with open("../NCD_data/ncd_data.json", 'w') as f:
            json.dump(data, f, indent=4)

    elif args.method == "ICDM":
        process_icdm(jsonl_file_paths, question_ids, args.num_students_per_file, args.sample_ratio)

    elif args.method == "few-shot":
        process_few_shot(jsonl_file_paths, "../process/combined_data.jsonl", "../process/skill_mapping.csv",
                         "../few-shot/datasets/moderate/fs_data/recordings.jsonl",
                         "../few-shot/datasets/moderate/fs_data/moderate_exercise_info.jsonl")

    print("[INFO] Process completed.")