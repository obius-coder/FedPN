
# Towards Federated Learning against Noisy Clients via CLIP-Guided Prototypes

## Abstract
Federated Noisy Labels Learning (FNLL) allows global model to be jointly trained on multiple clients with varying degrees of noisy labels while preserving privacy, and despite recent research advances, distinguishing between client clean and noisy samples is still tricky since the distribution of labels among clients is always both noisy and class-imbalanced, leading to the poor performance of existing FNLL methods. To address this problem, we propose a novel framework called FedPN, the first framework to utilize Contrastive Language-Image Pre-training (CLIP) for federated noisy labels tasks. Then, to achieve higher performance for the global model, we introduce an attention-based Prototype Adapter to identify more plausible local data for local model training, further improving training stability. We validate the effectiveness of FedPN by conducting extensive experiments on benchmark datasets under both Independently and Identically Distributed (IID) and Non-IID data partitions. The experimental results show that FedPN can effectively filter noisy samples from different clients, and compared with the state-of-the-art FNLL method, the FedPN achieves at most and at least 8.39% and 0.88% performance improvement in the case of highly heterogeneous noisy labels.



## Environments
Install [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). 

Install [conda latest](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

```bash
pip install -r requirements.txt # You may need to downgrade the torch using pip to match the CUDA version
```

## Training Parameters

### Federated Learning Stages:
- `--rounds1`: Number of communication rounds in the model warming-up stage
- `--rounds2`: Number of communication rounds in the usual training stage
- `--local_ep`: Number of local epochs for client training in each round
- `--frac`: Fraction of selected clients in each communication round

### Basic FL Settings:
- `--num_users`: Total number of clients in the federated system
- `--local_bs`: Local batch size for client training
- `--lr`: Learning rate for model optimization
- `--momentum`: Momentum parameter for SGD optimizer

### Model and Data:
- `--model`: Name of model architecture to use
- `--dataset`: Name of dataset to use
- `--pretrained`: Whether to use pre-trained model
- `--num_classes`: Number of classes in the dataset

### Data Distribution:
- `--iid`: Whether data is IID distributed among clients
- `--alpha_dirichlet`: Concentration parameter for Dirichlet distribution used in non-IID data partitioning


## Noise Settings
The system supports configurable noisy clients and label noise with the following parameters:

- `--level_n_system`: Fraction of noisy clients in the system (default: 0.4)
- `--level_n_lowerb`: Lower bound of noise level for noisy clients (default: 0.5)
- `--noise_type`: Type of label noise: `symmetric`, `pairflip`, or `pt` (default: 'pt')
- `--noise_rate`: Noise rate for each client's local data (default: 0.05)

### Noise Type Explanation:
- **symmetric**: Uniform label noise where each label has equal probability of being flipped to any other class
- **pairflip**: Pairwise flip noise where labels are only flipped between specific class pairs
- **pt**: Personalized noise that varies across different clients based on their characteristics

### Noise Level Control:
- For `pt` noise type, the system randomly selects `level_n_system * num_users` clients as noisy clients
- For each noisy client, the actual noise level is sampled between `[level_n_lowerb, 1.0]`
- For `symmetric` and `pairflip` noise types, the `noise_rate` parameter controls the probability of label corruption
- This allows simulating realistic scenarios where different clients have varying levels of data quality

## How to start simulating for FedPN
  
- Run evaluation FedPN (examples for MNIST):
    ```bash
    cd ./FedPN
    python main.py --dataset mnist --model cnn --rounds1 20 --rounds2 30 --num_users 50
    ```

- Example with symmetric noise:
    ```bash
    cd ./FedPN
    python main.py --dataset cifar10 --model resnet18 --rounds1 50 --rounds2 350 --num_users 50 \
    --noise_type symmetric --noise_rate 0.1
    ```

- Example with pairflip noise:
    ```bash
    cd ./FedPN
    python main.py --dataset cifar10 --model resnet18 --rounds1 50 --rounds2 350 --num_users 50 \
    --noise_type pairflip --noise_rate 0.15
    ```

- Example with pt noise (noise_rate not used):
    ```bash
    cd ./FedPN
    python main.py --dataset cifar10 --model resnet18 --rounds1 50 --rounds2 350 --num_users 50 \
    --level_n_system 0.6 --level_n_lowerb 0.5 --noise_type pt
    ```

If you have any questions, feel free to leave a message or email us.
