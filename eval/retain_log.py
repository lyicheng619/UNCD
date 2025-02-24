import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import argparse
import gc  # Garbage collector

def load_state(state_file):
    try:
        with open(state_file, 'r') as file:
            state = json.load(file)
            return {
                "total_questions_processed": state.get("total_questions_processed", 0),
                "correct_answers": state.get("correct_answers", 0),
                "total_questions": state.get("total_questions", 0),
            }
    except (FileNotFoundError, json.JSONDecodeError):
        return {"total_questions_processed": 0, "correct_answers": 0, "total_questions": 0}

def save_state(state_file, state):
    with open(state_file, 'w') as file:
        json.dump(state, file)

def answer_mcq(stem, choices, model, tokenizer, device):
    log_probs = []
    
    with torch.no_grad():  # Apply globally for memory efficiency
        inputs = tokenizer([stem + " " + choice for choice in choices], return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        input_ids = inputs['input_ids']

        for i in range(len(choices)):
            log_prob = logits[i, :-1, :].gather(1, input_ids[i, 1:].unsqueeze(-1)).squeeze(-1).sum().item()
            log_probs.append(log_prob)

        del inputs
        torch.cuda.empty_cache()

    return log_probs.index(max(log_probs))

def process_questions(input_file, output_file, state_file, model_name, device_number=0):
    state = load_state(state_file)
    total_questions_processed = state["total_questions_processed"]
    correct_answers = state["correct_answers"]
    total_questions = state["total_questions"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half()
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(input_file, 'r') as in_file, open(output_file, 'a') as out_file:
        questions_skipped = 0
        for line in in_file:
            if questions_skipped < total_questions_processed:
                questions_skipped += 1
                continue

            data = json.loads(line)
            stem = data['stem']
            choices = data['choices']
            answer_index = data["ans"]
            total_questions += 1

            best_choice = answer_mcq(stem, choices, model, tokenizer, device)
            correct = 1 if best_choice == answer_index else 0
            correct_answers += correct

            data["correct"] = correct
            json.dump(data, out_file)
            out_file.write('\n')
            out_file.flush()  # Ensure immediate write to file

            state = {
                "total_questions_processed": total_questions,
                "correct_answers": correct_answers,
                "total_questions": total_questions
            }
            save_state(state_file, state)

            if total_questions % 10 == 0:
                running_accuracy = (correct_answers / total_questions) * 100
                print(f"Processed {total_questions} questions with an accuracy of {running_accuracy:.2f}%.")

            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCQ Answering Script using a Transformer Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("model_id", type=str, help="Unique identifier for the model (used for logging directory).")
    parser.add_argument("device_number", type=int, nargs="?", default=0, help="CUDA device number to use (default: 0).")

    args = parser.parse_args()

    # Create log directory if it doesn't exist
    log_dir = f"{args.model_id}_retain"
    os.makedirs(log_dir, exist_ok=True)

    # Define paths
    input_file = "../data/retain_eval.jsonl"
    output_file = os.path.join(log_dir, "output.jsonl")
    state_file = os.path.join(log_dir, "state.json")

    # Run the process
    process_questions(input_file, output_file, state_file, "model_path", device_number=args.device_number)