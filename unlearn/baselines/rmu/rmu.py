import os
import datetime

import numpy as np
import torch
from transformers import AdamW
import tqdm as tqdm

from rmu.utils import load_model, get_params, forward_with_cache, get_data

def save_checkpoint(model, tokenizer, optimizer, args, batch_idx, epoch):
    if args.output_dir:
        base_directory = args.output_dir
    else:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base_directory = f"models/{args.model_name_or_path}_{date_str}"

    checkpoint_directory = os.path.join(base_directory, f"checkpoint-{batch_idx + 1}")
    os.makedirs(checkpoint_directory, exist_ok=True)
    model.save_pretrained(checkpoint_directory)
    tokenizer.save_pretrained(checkpoint_directory)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_directory, "optimizer.pt"))

    # Save additional state information
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx+1
    }
    torch.save(state, os.path.join(checkpoint_directory, "training_state.pt"))

    print(f"Checkpoint saved at {checkpoint_directory}")
    return checkpoint_directory

def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k,v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    start_epoch = 0
    start_batch_idx = 0

    if args.checkpoint_path:
        optimizer_state_path = os.path.join(args.checkpoint_path, "optimizer.pt")
        optimizer.load_state_dict(torch.load(optimizer_state_path))

        training_state_path = os.path.join(args.checkpoint_path, "training_state.pt")
        training_state = torch.load(training_state_path)
        start_epoch = training_state['epoch']
        start_batch_idx = training_state['batch_idx']
        print(start_batch_idx)

    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list])
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    # Compute checkpoint interval
    checkpoint_interval = num_batches // (args.n_step - 1) if args.n_step > 1 else num_batches

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            pbar.n = start_batch_idx
            pbar.refresh()
            for idx in range(start_batch_idx, num_batches):
                # Save checkpoint at computed intervals and at the final step
                if (idx + 1) % checkpoint_interval == 0 or idx + 1 == num_batches:
                    save_path = save_checkpoint(updated_model, tokenizer, optimizer, args, idx, epoch)
                    print(f"Checkpoint saved to {save_path}")

                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 2048 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= args.alpha[topic_idx]

                # Update model
                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                
                if args.verbose:
                    frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                    unlearn_cosine = torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                    
                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(f"Topic {topic_idx} updated_forget_activations.norm=", torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_forget_activations.norm=", torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} updated_retain_activations.norm=", torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_retain_activations.norm=", torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())

                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    # Save final model
    path = args.output_dir if args.output_dir else f"models/{args.model_name_or_path}_alpha-{args.alpha}_batches-{num_batches}_layer-{args.layer_id}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_step", type=int, default=3, help="Number of checkpoints (including final model)")
    # (Other arguments remain the same)
    
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frozen_model, tokenizer = load_model(args.model_name_or_path, args.checkpoint_path)
    updated_model, tokenizer = load_model(args.model_name_or_path, args.checkpoint_path)
    forget_data_list, retain_data_list = get_data(args.forget_corpora, args.retain_corpora, args.min_len, args.max_len, args.batch_size)

    run_rmu(updated_model, frozen_model, tokenizer, forget_data_list, retain_data_list, args)