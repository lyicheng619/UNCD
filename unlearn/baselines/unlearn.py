import sys
import pathlib
import argparse
from os.path import basename, dirname, join as pathjoin

from transformers import AutoModelForCausalLM
import torch

# Import baselines
from baselines import it_unlearn, tv_unlearn, finetune

# Import RMU components from baselines/rmu
from baselines.rmu.rmu import run_rmu  # Import run_rmu from baselines/rmu
from baselines.rmu.utils import load_model as rmu_load_model, get_data  # Import RMU utilities

BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)

# Load model function
def load_model(model_dir: str, **kwargs) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        **kwargs
    )

# Compare two models
def compare(model1, model2) -> bool:
    dict1, dict2 = model1.state_dict(), model2.state_dict()
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True

# Main function
def main():
    args = get_args()

    if args.algo == 'kn':
        raise NotImplementedError()

    elif args.algo == 'tv':
        ft_model_dir = pathjoin(dirname(args.out_dir), basename(args.out_dir) + "_ft")
        finetune(
            args.model_dir, args.data_file, ft_model_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir
        )
        tv_unlearn(
            args.model_dir, args.out_dir,
            some_pt_model_dir=args.model_dir,
            some_ft_model_dir=ft_model_dir,
            alpha=args.alpha
        )

        model1 = load_model(args.model_dir)
        model2 = load_model(args.out_dir)
        print(compare(model1, model2))

    elif args.algo == 'rmu':
        # Load frozen and updated models along with tokenizer
        frozen_model, tokenizer = rmu_load_model(args.model_name_or_path, args.checkpoint_path)
        updated_model, tokenizer = rmu_load_model(args.model_name_or_path, args.checkpoint_path)

        # Load forget and retain datasets
        forget_data_list, retain_data_list = get_data(
            args.forget_corpora,
            args.retain_corpora,
            args.min_len,
            args.max_len,
            args.batch_size
        )

        # Run RMU algorithm
        run_rmu(
            updated_model,
            frozen_model,
            tokenizer,
            forget_data_list,
            retain_data_list,
            args
        )

    else:
        it_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )

    return

# Argument parsing function
def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")

    parser.add_argument('--algo', type=str, required=True, help="Algorithm to use: ga, gd, tv, rmu, etc.")

    # Model paths
    parser.add_argument('--model_dir', type=str, help="Path to the target model's Hugging Face directory.")
    parser.add_argument('--tokenizer_dir', type=str, default=None, help="Path to the tokenizer's HF directory.")
    parser.add_argument('--data_file', type=str, help="Path to the forget set file.")
    parser.add_argument('--out_dir', type=str, help="Path to the output model's directory.")
    parser.add_argument('--max_len', type=int, default=4096, help="Max length of input sequences.")
    parser.add_argument('--resume_from_checkpoint', action='store_true', help="Resume training from checkpoint.")

    # Gradient-based methods
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument('--retain_data_file', type=str, default=None, help="Required if algo is gradient difference (gd).")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate for ga, gd, tv.")
    parser.add_argument('--epochs', type=int, default=5, help="Epochs for ga, gd, tv.")
    parser.add_argument('--alpha', type=float, default=1.0, help="Scaling coefficient for task vector (tv).")

    if "--algo" in sys.argv and "rmu" in sys.argv:
        # RMU-specific arguments
        parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta")
        parser.add_argument("--module_str", type=str, default="{model_name}.model.layers[{layer_id}]")
        parser.add_argument("--output_dir", type=str, default=None)
        parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext", help="Comma-separated corpora to retain.")
        parser.add_argument("--forget_corpora", type=str, default="bio-forget-corpus,cyber-forget-corpus", help="Comma-separated corpora to forget.")
        parser.add_argument("--steering_coeffs", type=str, default="20,20", help="Steering vector weight for each topic.")
        parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a saved checkpoint.")
        parser.add_argument("--min_len", type=int, default=0)
        parser.add_argument("--max_len", type=int, default=2000)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--max_num_batches", type=int, default=80)
        parser.add_argument("--layer_id", type=int, default=7, help="Layer to unlearn.")
        parser.add_argument("--layer_ids", type=str, default="5,6,7", help="Layers to update.")
        parser.add_argument("--param_ids", type=str, default="6", help="Parameters to update.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")

    args = parser.parse_args()

    # Additional processing for RMU arguments
    if args.algo == 'rmu':
        args.retain_corpora = args.retain_corpora.split(",")
        args.forget_corpora = args.forget_corpora.split(",")
        args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
        args.alpha = [float(c) for c in args.alpha.split(",")]
        args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
        args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]

    return args

if __name__ == '__main__':
    main()