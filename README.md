# Deep Reinforcement Learning for Image Enhancement (DQN + Vision Transformer)

This project implements an image enhancement pipeline using Deep Q-Networks (DQN) with ***TensorFlow Agents*** and ***Vision Transformer*** (ViT) embeddings from HuggingFace Transformers.

The agent learns to select the best adaptive histogram equalization tile size to maximize perceptual quality, measured by EME (Measure of Enhancement), while penalizing computational cost.

## Key Features

- Custom RL environment (ImageEnhanceEnv) for image enhancement.
- State representation: Vision Transformer (ViT) embeddings of input images.
- Actions: Different tile_size values for Adaptive Histogram Equalization (CLAHE).
- Reward: EME (enhancement quality) – λ × computation time.
- Comparison against Global Histogram Equalization and CLAHE baseline.
- Training with DQN agent (TF-Agents).
- Policy saving/loading, checkpointing, and TensorBoard logging.

## Project Structure
. <br>
├── main.py                  # Main training & evaluation script <br>
├── checkpoint/              # Checkpoints for agent & replay buffer <br>
├── policy_eval/             # Saved evaluation (greedy) policy <br>
├── policy_collect/          # Saved collect policy (for resuming training) <br>
├── logs/                    # TensorBoard training logs <br>
└── coco/                    # Automatically downloaded COCO val2017 dataset

## Requirements
### Python Environment

Recommended: Python 3.9+ with GPU-enabled PyTorch and TensorFlow.

## Install Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 <br>
pip install tensorflow tensorflow-probability tensorflow-agents <br>
pip install transformers opencv-python matplotlib

## Dataset

The project uses the COCO val2017 dataset (5,000 images). <br>
It will be automatically downloaded and extracted to ./coco/val2017 when you first run the script. <br>
 <br>
If you prefer to download manually: <br>
wget http://images.cocodataset.org/zips/val2017.zip -O coco/val2017.zip <br>
unzip coco/val2017.zip -d coco/

## Running the Project
### Train the DQN Agent
python main.py <br>
 <br>
 <br>
The script will: <br>
- Download and preprocess COCO images (grayscale + ViT embeddings).
- Initialize the RL environment.
- Train a DQN agent for ~1500 iterations.
- Save checkpoints, policies, and logs.

## Monitor Training

Use TensorBoard to monitor rewards and loss: <br>
tensorboard --logdir logs/fit

## Evaluation & Comparison

The script evaluates on test images, showing comparisons between:
- Original
- DRL-enhanced (DQN-selected tile size)
- CLAHE (fixed parameters)
- Global Histogram Equalization

Example output: <br>
📷 Side-by-side plots with EME values per method.

## Saving & Loading Policies

- Evaluation Policy (for inference): saved to policy_eval/
- Collect Policy (for resuming training): saved to policy_collect/
- You can later load them using TF-Agents’ PolicySaver.

## References

- TF-Agents: DQN Agent
- HuggingFace Transformers: Vision Transformer
- COCO Dataset: http://cocodataset.org
