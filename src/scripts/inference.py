import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import os
from peft import PeftModel

# Add parent directory to path to import data_loader
sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import LotkaVolterraDataset, encode, decode


def list_checkpoints(checkpoints_dir="fine_tuned_models"):
    """List available LoRA checkpoints."""
    if not os.path.exists(checkpoints_dir):
        print(f"No checkpoints directory found: {checkpoints_dir}")
        return []
    
    checkpoints = [d for d in os.listdir(checkpoints_dir) 
                  if os.path.isdir(os.path.join(checkpoints_dir, d))]
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoints_dir}")
        return []
    
    print("Available LoRA checkpoints:")
    for i, checkpoint in enumerate(sorted(checkpoints), 1):
        print(f"  {i}. {checkpoint}")
    
    return sorted(checkpoints)


def load_model(model_name, checkpoint_path=None, device="auto"):
    """Load base model or LoRA fine-tuned model."""
    print(f"Loading tokenizer and base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set device map based on available hardware
    if device == "auto":
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        device_map = device
        torch_dtype = torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        device_map=device_map
    )
    
    if checkpoint_path:
        if checkpoint_path == "latest":
            checkpoints = list_checkpoints()
            if not checkpoints:
                print("No checkpoints available, using base model")
                return tokenizer, base_model, "base"
            checkpoint_path = f"fine_tuned_models/{checkpoints[-1]}"
        elif not checkpoint_path.startswith("fine_tuned_models/"):
            checkpoint_path = f"fine_tuned_models/{checkpoint_path}"
        
        if os.path.exists(checkpoint_path):
            print(f"Loading LoRA adapter from: {checkpoint_path}")
            try:
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                print("✓ LoRA model loaded successfully!")
                return tokenizer, model, f"LoRA ({os.path.basename(checkpoint_path)})"
            except Exception as e:
                print(f"✗ Error loading LoRA model: {e}")
                print("Falling back to base model")
                return tokenizer, base_model, "base (LoRA load failed)"
        else:
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints:")
            list_checkpoints()
            print("Using base model instead")
    
    return tokenizer, base_model, "base"


def run_inference(model, tokenizer, dataset, device):
    """Run inference on the first example from dataset."""
    model.eval()
    
    def preprocess_function(examples):
        return tokenizer(examples["text"],
                         truncation=True,
                         padding=False,
                         max_length=tokenizer.model_max_length)
    
    # Prepare dataset
    texts = encode(dataset.trajectories)
    hf_dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    
    # Run inference on first example
    example = tokenized_dataset[0]
    print(f"Input sequence length: {len(example['input_ids'])} tokens")
    
    input_ids = torch.tensor(example["input_ids"])[None, :]
    attention_mask = torch.tensor(example["attention_mask"])[None, :]
    
    # Move to appropriate device if not using device_map
    if isinstance(device, str) and device != "auto":
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    
    print("Generating predictions...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode results
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    original_text = texts[0]
    
    print(f"Generated text length: {len(generated_text)} characters")
    print(f"Original text length: {len(original_text)} characters")
    
    return original_text, generated_text


def plot_results(original_text, generated_text, model_type):
    """Plot comparison between original and generated trajectories."""

    original_trajectory = decode([original_text])[0]
    generated_trajectory = decode([generated_text])[0]
    
    plt.figure(figsize=(14, 8))
    
    # Plot trajectories over time
    plt.subplot(2, 1, 1)
    plt.plot(original_trajectory[:, 0], label="Original Prey", marker='o', alpha=0.7)
    plt.plot(original_trajectory[:, 1], label="Original Predator", marker='o', alpha=0.7)
    plt.plot(generated_trajectory[:, 0], label="Generated Prey", marker='x', alpha=0.7)
    plt.plot(generated_trajectory[:, 1], label="Generated Predator", marker='x', alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title(f"Time Series Comparison ({model_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot phase space (Prey vs Predator)
    plt.subplot(2, 1, 2)
    plt.plot(original_trajectory[:, 0], original_trajectory[:, 1], 
            label="Original Phase Space", marker='o', alpha=0.7)
    plt.plot(generated_trajectory[:, 0], generated_trajectory[:, 1], 
            label="Generated Phase Space", marker='x', alpha=0.7)
    plt.xlabel("Prey Population")
    plt.ylabel("Predator Population")
    plt.title("Phase Space Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen base or LoRA fine-tuned model")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="LoRA checkpoint name or 'latest' (default: use base model)")
    parser.add_argument("--list-checkpoints", action="store_true",
                       help="List available checkpoints and exit")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'auto', 'cpu', 'cuda' (default: auto)")
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        list_checkpoints()
        return
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Load model
    tokenizer, model, model_type = load_model(model_name, args.checkpoint, args.device)
    print(f"Using model: {model_type}")
    
    # Load dataset
    print("Loading Lotka-Volterra dataset...")
    dataset = LotkaVolterraDataset()
    print(f"Dataset loaded: {dataset.trajectories.shape[0]} trajectories")
    
    # Run inference
    original_text, generated_text = run_inference(model, tokenizer, dataset, args.device)
    
    # Plot results
    plot_results(original_text, generated_text, model_type)


if __name__ == "__main__":
    main()
