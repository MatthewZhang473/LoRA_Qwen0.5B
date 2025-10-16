import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import data_loader
sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import LotkaVolterraDataset, encode, decode


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    model.eval()
    
    
    def preprocess_function(examples):
        return tokenizer(examples["text"],
                         truncation=True,
                         padding=False,
                         max_length=tokenizer.model_max_length)
    
    # Load dataset
    raw_dataset = LotkaVolterraDataset()
    trajectories = raw_dataset.trajectories
    texts = encode(trajectories)
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    
    # run inference on the first example
    example = tokenized_dataset[0]
    print(f"Number of tokens of the example sequence: {len(example['input_ids'])}")
    input_ids = torch.tensor(example["input_ids"])[None, :]
    attention_mask = torch.tensor(example["attention_mask"])[None, :]
    
    generated_ids = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_new_tokens=1000)
    
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # plot
    decoded_trajectory = decode([decoded_output])[0]  # Decode the generated text
    original_trajectory = decode([texts[0]])[0]  # Decode the original text

    plt.figure(figsize=(12, 6))
    plt.plot(original_trajectory[:, 0], label="Original Prey", marker='o')
    plt.plot(original_trajectory[:, 1], label="Original Predator", marker='o')
    plt.plot(decoded_trajectory[:, 0], label="Decoded Prey", marker='x')
    plt.plot(decoded_trajectory[:, 1], label="Decoded Predator", marker='x')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Original vs Decoded Trajectory Over Time")
    plt.legend()
    plt.grid()
    plt.show()
        
if __name__ == "__main__":
    main()

        