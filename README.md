# Neural Network Quantization and Compression

A comprehensive framework for neural network quantization and compression, implementing various quantization methods and optimization techniques to reduce model size while maintaining performance.

## Overview

This project provides tools for neural network compression through quantization and optimization. It implements multiple quantization methods and uses evolutionary algorithms to find optimal compression parameters. The framework supports various neural network architectures and datasets.

## Features

- Multiple quantization methods:
  - Uniform quantization
  - Linear quantization
  - Forgy quantization
  - Density-based quantization
- Huffman encoding for additional compression
- Support for various neural network architectures (ResNet, etc.)
- Multi-objective optimization using NSGA-II
- Support for multiple datasets (CIFAR-10, ImageNet, etc.)
- Visualization of compression results and Pareto fronts

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- pymoo
- Matplotlib
- scikit-learn
- SciencePlots (optional, for publication-quality plots)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/moea-compression.git
cd moea-compression
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Train DNNs

```bash
python trainer.py \
    --dataset cifar10 \
    --arch resnet18 \
    --save_dir "./checkpoints/cifar10/resnet18/" \
    --epochs 182 \
    --seed 6
```

### Basic Quantization

To run the quantization process on a pre-trained model:

```bash
python quantizer.py \
    --dataset cifar10 \
    --arch resnet18 \
    --model_path "./checkpoints/cifar10/resnet18/" \
    --save_dir "./logs/cifar10/resnet18/ws/seed6/" \
    --steps 10 \
    --seed 6
```

### Iterative Merger

To run the iterative merge on process on a pre-trained model:

```bash
python merger.py \
    --dataset cifar10 \
    --arch resnet18 \
    --model_path "./checkpoints/cifar10/resnet18/" \
    --save_dir "./logs/cifar10/resnet18/ws/seed6/" \
    --seed 6
```

### Command Line Arguments

- `--dataset`: Dataset to use (cifar10, imagenet, etc.)
- `--arch`: Model architecture (resnet18, etc.)
- `--model_path`: Path to the pre-trained model
- `--save_dir`: Directory to save results
- `--steps`: Number of optimization steps
- `--seed`: Random seed for reproducibility
- `--input_size`: Size of input images
- `--num_workers`: Number of data loading workers
- `--num_classes`: Number of classes in the dataset

## Project Structure

- `quantizer.py`: Main quantization and optimization implementation
- `compression/`: Core compression techniques
  - `quantization.py`: Various quantization methods
  - `huffman_encoding.py`: Huffman encoding implementation
  - `pruning.py`: Pruning utilities
- `models/`: Neural network model definitions
- `utils.py`: Utility functions
- `arg_parser.py`: Command line argument parsing
- `dataset.py`: Dataset handling and preprocessing

## Quantization Methods

The framework implements several quantization methods:

1. **Uniform Quantization**: Divides the weight range into equal intervals
2. **Linear Quantization**: Uses linear scaling for quantization
3. **Forgy Quantization**: Uses k-means clustering for quantization
4. **Density Quantization**: Considers weight distribution density

## License

This project is licensed under the terms of the included LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{moea-compression,
  author = {Rasa Khosrowshahli, Shahryar Rahnamayan, Beatrice Ombuki-Berman},
  title = {Neural Network Quantization and Compression Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rkhosrowshahli/moea-compression}
}
```
