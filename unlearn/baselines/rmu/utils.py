
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)

################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path, checkpoint_path=None):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if checkpoint_path:
        print(f"Loading model and tokenizer from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False
        )
    else:
        print(f"Loading model and tokenizer from model path: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False
        )

    # Configure tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer



def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(file_path):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                raw_text = line.strip()  # Just read the raw text
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        # Batch the data
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    # Since forget_corpora and retain_corpora are lists of file paths, iterate over them
    forget_data = [get_dataset(path) for path in forget_corpora]
    retain_data = [get_dataset(path) for path in retain_corpora]

    return forget_data, retain_data

