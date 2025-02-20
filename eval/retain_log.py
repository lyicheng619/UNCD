import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import argparse

def load_state(state_file):
    try:
        with open(state_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"total_questions_processed": 0, "correct_answers": 0, "total_questions": 0}

def save_state(state_file, state):
    with open(state_file, 'w') as file:
        json.dump(state, file)

def answer_mcq(stem, choices, model, tokenizer, device):
    inputs = [tokenizer(stem + " " + choice, return_tensors="pt").to(device) for choice in choices]
    log_probs = []
    for input_data in inputs:
        with torch.no_grad():
            outputs = model(**input_data)
            logits = outputs.logits
            input_ids = input_data['input_ids']
            log_prob = logits[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(1).item()
            log_probs.append(log_prob)
        
        del input_data
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
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some questions.")
    parser.add_argument("device_number", type=int, help="CUDA device number to use")
    args = parser.parse_args()

    # Example usage
    process_questions(
        "data_path",
        "output.jsonl",
        "state.json",
        "model_path",
        device_number=args.device_number
    )