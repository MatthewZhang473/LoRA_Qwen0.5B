# LoRA Fine-tuning for Time Series Prediction with Qwen 0.5B

This repository demonstrates how to use **LoRA (Low-Rank Adaptation)** to fine-tune the Qwen 0.5B language model for time series prediction tasks. The project showcases an efficient approach to adapt large language models for numerical sequence forecasting while maintaining computational efficiency.

## 🎯 Project Overview

This project explores the application of Large Language Models (LLMs) to time series forecasting by:
- Fine-tuning Qwen 0.5B model using LoRA technique
- Converting time series data into text format for LLM processing
- Training on Google Colab with A100 GPU for optimal performance
- Demonstrating efficient parameter updating with LoRA's low-rank matrices

## 🔧 Key Features

- **LoRA Fine-tuning**: Efficient adaptation using low-rank decomposition
- **Time Series to Text**: Novel approach converting numerical sequences to text tokens
- **GPU Optimized**: Training scripts optimized for A100 GPU on Google Colab
- **Memory Efficient**: LoRA reduces trainable parameters by ~99%
- **Modular Design**: Clean, reusable code structure

## 📁 Repository Structure

```
LoRA_Qwen0.5B/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train_lora.py            # Main training script
├── data/                    # Time series datasets
├── models/                  # Model checkpoints and configurations
├── notebooks/               # Jupyter notebooks for Colab
└── utils/                   # Utility functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: A100 on Google Colab)
- Sufficient RAM (16GB+ recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LoRA_Qwen0.5B
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Google Colab users:
```python
!git clone <repository-url>
%cd LoRA_Qwen0.5B
!pip install -r requirements.txt
```

## 💡 Usage

### Local Training

```bash
python train_lora.py --config config/default.yaml
```

### Google Colab Training

1. Open the provided Colab notebook
2. Mount Google Drive (optional, for data storage)
3. Run the training cells
4. Monitor training progress with built-in visualizations

### Key Parameters

- `--model_name`: Base model (default: Qwen/Qwen2.5-0.5B)
- `--lora_rank`: LoRA rank parameter (default: 16)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--batch_size`: Training batch size (default: 8)
- `--max_steps`: Maximum training steps (default: 1000)

## 📊 Model Architecture

### LoRA Configuration
- **Rank (r)**: 16 (adjustable)
- **Alpha**: 32
- **Target Modules**: Query, Value projections
- **Dropout**: 0.1

### Time Series Processing
- Sequences converted to text format
- Sliding window approach for training samples
- Normalization and scaling applied

## 📈 Results

The LoRA fine-tuned model achieves:
- Significant reduction in training time (vs full fine-tuning)
- Comparable prediction accuracy
- 99% reduction in trainable parameters
- Efficient memory usage on consumer GPUs

## 🛠️ Customization

### Modifying LoRA Parameters
Edit the LoRA configuration in your training script:
```python
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Alpha parameter
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.