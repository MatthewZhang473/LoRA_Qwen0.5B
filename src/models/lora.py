"""
Low-Rank Adaptation (LoRA) implementation for efficient fine-tuning of large language models.

This module provides a complete implementation of the LoRA method (Hu et al., 2021)
for parameter-efficient fine-tuning, including:
- LoRA linear layer implementation that injects trainable rank-decomposition matrices
- Utilities to apply LoRA to specific model components (attention layers)
- Functions for saving and loading LoRA weights
- Training and evaluation helpers for time series forecasting

The implementation focuses on Qwen2.5 models but can be adapted for other transformer architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import argparse
import os
import json
from pathlib import Path
import logging
import math
import sys
from datetime import datetime
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import text_to_numeric, numeric_to_text, load_data
from src.models.qwen import load_qwen

from utils.lora_flop_tracker import LoRAFLOPTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(project_root) / "results"  # Use the project_root variable
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "finetune_figures"
FIGURES_DIR.mkdir(exist_ok=True)


# LoRA implementation
class LoRALinear(nn.Module):
    """
    LoRA implementation for Linear layers:
    y = W*x + b + (A*B)*x * (alpha/r)
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None, dropout: float = 0.0):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        
        # Store original layer and freeze it
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        
        # Get dimensions
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # LoRA parameters
        self.r = r
        self.alpha = alpha if alpha is not None else r
        
        # Get device from original layer
        device = original_linear.weight.device
        
        # Define LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device))
        
        # Optional dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A matrix (B is initialized to zeros)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        # Original output
        base_output = self.original_linear(x)
        
        # LoRA path with dropout
        lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        
        # Combine with scaling factor
        return base_output + lora_output * (self.alpha / self.r)


def apply_lora_to_model(model, r=8, alpha=16, dropout=0.0, target_modules=None):
    """
    Apply LoRA to specific modules in the model.
    
    Args:
        model: The model to add LoRA to
        r: LoRA rank
        alpha: LoRA alpha (scaling)
        dropout: Dropout rate for LoRA layers
        target_modules: List of module types to apply LoRA to (if None, apply to Q and V projections only)
    
    Returns:
        Modified model
    """
    # If no target modules specified, default to Q and V projections
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # Track parameters to train
    trainable_params = []
    
    # Apply LoRA to attention modules
    for layer in model.model.layers:
        for name, module in layer.self_attn.named_modules():
            # Check if this is a target module
            if name in target_modules and isinstance(module, nn.Linear):
                logger.info(f"Applying LoRA (r={r}, alpha={alpha}) to {name}")
                
                # Replace with LoRA module
                if name == "q_proj":
                    layer.self_attn.q_proj = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    trainable_params.extend([p for p in layer.self_attn.q_proj.parameters() if p.requires_grad])
                elif name == "v_proj":
                    layer.self_attn.v_proj = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    trainable_params.extend([p for p in layer.self_attn.v_proj.parameters() if p.requires_grad])
    
    # Add bias parameter if it exists
    if model.lm_head.bias is not None and model.lm_head.bias.requires_grad:
        trainable_params.append(model.lm_head.bias)
    
    return model, trainable_params


def process_sequences(texts, tokenizer, max_length=512, stride=256, add_eos=False):
    """
    Process text sequences into tokenized chunks for training.
    
    Args:
        texts: List of text sequences
        tokenizer: Tokenizer to use
        max_length: Maximum length of each chunk
        stride: Stride between consecutive chunks
        add_eos: Whether to add EOS token
        
    Returns:
        List of tokenized sequences
    """
    all_input_ids = []
    all_labels = []
    
    for text in texts:
        # Apply tokenization scheme
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        
        # If sequence is too long, create chunks with sliding window
        if len(seq_ids) > max_length:
            # Create sliding windows
            for i in range(0, len(seq_ids) - max_length + 1, stride):
                chunk = seq_ids[i:i + max_length]
                all_input_ids.append(chunk)
                all_labels.append(chunk.clone())  # For autoregressive loss
        else:
            # Pad sequence if it's shorter than max_length
            if len(seq_ids) < max_length:
                padding = torch.full((max_length - len(seq_ids),), tokenizer.pad_token_id)
                input_ids = torch.cat([seq_ids, padding])
                
                # # Create attention mask
                attention_mask = torch.cat([
                    torch.ones(len(seq_ids), dtype=torch.long),
                    torch.zeros(max_length - len(seq_ids), dtype=torch.long)
                ])
                
                # For shorter sequences, only use the actual sequence as labels
                labels = torch.cat([seq_ids, torch.full((max_length - len(seq_ids),), -100)])
            else:
                input_ids = seq_ids
                attention_mask = torch.ones(max_length, dtype=torch.long)
                labels = seq_ids.clone()
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
    
    return torch.stack(all_input_ids), torch.stack(all_labels)


def get_grad_norm(model):
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm




def save_lora_model(model, output_path):
    """
    Save LoRA weights and lm_head bias.
    
    Args:
        model: Model with LoRA layers
        output_path: Path to save the model weights
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Extract and save LoRA weights
    lora_state_dict = {}
    lora_r = None
    lora_alpha = None
    
    # Save LoRA modules
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()
            
            # Capture parameters from the first LoRA module
            if lora_r is None:
                lora_r = module.r
                lora_alpha = module.alpha
    
    # Save lm_head bias if it exists and is trainable
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
        lora_state_dict["lm_head.bias"] = model.lm_head.bias.data.cpu()
    
    # Save weights
    torch.save(lora_state_dict, os.path.join(output_path, "lora_weights.pt"))
    
    # Save config
    config = {
        "model_type": "Qwen2.5-0.5B-Instruct",
        "lora_applied_modules": ["q_proj", "v_proj"],
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    }
    
    with open(os.path.join(output_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    



def load_lora_weights(model, weights_path):
    """
    Load LoRA weights and lm_head bias into a model.
    
    Args:
        model: Model with LoRA layers
        weights_path: Path to the saved weights
        
    Returns:
        Model with loaded weights
    """
    lora_state_dict = torch.load(weights_path, map_location="cpu")
    
    # Load LoRA module weights
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data.copy_(lora_state_dict[f"{name}.lora_A"])
            if f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data.copy_(lora_state_dict[f"{name}.lora_B"])
    
    # Load lm_head bias if available
    if "lm_head.bias" in lora_state_dict and hasattr(model, 'lm_head'):
        if model.lm_head.bias is None:
            # If bias doesn't exist, create it
            model.lm_head.bias = nn.Parameter(lora_state_dict["lm_head.bias"].to(model.device))
        else:
            # If bias exists, update it
            model.lm_head.bias.data.copy_(lora_state_dict["lm_head.bias"].to(model.device))
    
    return model


def load_validation_data_from_file(file_path, input_steps=50, forecast_steps=50, num_samples=50, random_seed=42):
    """
    Load validation data from a text file containing time series.
    
    Args:
        file_path: Path to the validation data file
        input_steps: Number of steps to use as input
        forecast_steps: Number of steps to forecast
        num_samples: Maximum number of samples to load
        random_seed: Random seed for selecting samples consistently
        
    Returns:
        List of dictionaries with input_sequence and ground_truth
    """
    validation_data = []
    
    try:
        # Read all lines from the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Set the random seed for reproducibility
        np.random.seed(random_seed)
        
        # Process eligible lines
        eligible_lines = []
        
        for i, line in enumerate(lines):
            # Clean and split the line
            full_sequence = line.strip()
            timesteps = full_sequence.split(';')
            
            # Skip if too short
            if len(timesteps) < input_steps + forecast_steps:
                logger.warning(f"Sequence {i} is too short ({len(timesteps)} steps), skipping")
                continue
            
            eligible_lines.append((i, full_sequence, timesteps))
        
        # Randomly select samples
        if len(eligible_lines) > num_samples:
            selected_indices = np.random.choice(len(eligible_lines), num_samples, replace=False)
            selected_lines = [eligible_lines[i] for i in selected_indices]
        else:
            selected_lines = eligible_lines
        
        # Process selected lines
        for i, full_sequence, timesteps in selected_lines:
            # Take only the first input_steps for input
            input_sequence = ';'.join(timesteps[:input_steps])
            
            # The rest is ground truth (up to forecast_steps)
            ground_truth_steps = timesteps[input_steps:input_steps+forecast_steps]
            ground_truth = ';'.join(ground_truth_steps)
            
            if input_sequence and ground_truth:
                validation_data.append({
                    "input_sequence": input_sequence,
                    "ground_truth": ground_truth,
                    "original_idx": i
                })
        
        logger.info(f"Loaded {len(validation_data)} validation sequences from {file_path} (random seed: {random_seed})")
        
    except Exception as e:
        logger.error(f"Error loading validation data from {file_path}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return validation_data


