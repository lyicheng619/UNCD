import os
from utils import write_json, write_text
import json

def load_jsonl(file_path):
    """ Load JSONL file and return a list of text entries. """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

os.makedirs('data', exist_ok=True)
forget_file_path = '../data/forget.jsonl'
retain_file_path = '../data/retain.jsonl'

forget_data = load_jsonl(forget_file_path)
retain_data = load_jsonl(retain_file_path)

write_json(forget_data, "data/forget.json")
write_json(retain_data, "data/retain.json")

write_text("\n\n".join(forget_data), "data/forget.txt")
write_text("\n\n".join(retain_data), "data/retain.txt")
