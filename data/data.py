import json

# Define the input and output file names
input_file = "retain_eval.jsonl"
output_file = "output.jsonl"

# Define the desired key order
desired_order = ["stem", "choices", "ans", "sub-domain", "domain"]

# Open the input JSONL file for reading and output JSONL file for writing
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # Load JSON object from the line
        data = json.loads(line.strip())

        # Rename keys
        if "Domain" in data:
            data["domain"] = data.pop("Domain")
        if "SubDomain" in data:
            data["sub-domain"] = data.pop("SubDomain")

        # Reorder keys
        ordered_data = {key: data[key] for key in desired_order if key in data}

        # Write the modified JSON object to the output file
        outfile.write(json.dumps(ordered_data, ensure_ascii=False) + "\n")

print(f"Processed JSONL file saved as '{output_file}'")