# MOEA Compression

A multi-objective evolutionary algorithm (MOEA) based approach for neural network compression, combining pruning and quantization techniques to optimize model size and performance.

## Overview

This project implements a multi-objective optimization approach for neural network compression using NSGA-II (Non-dominated Sorting Genetic Algorithm II). It combines two main compression techniques:

1. **Weight-sharing Quantization**: Reduces the precision of weights while maintaining model performance

The optimization process aims to find the optimal trade-off between model size and accuracy.

## Features

- Multi-objective optimization using NSGA-II
- Support for various neural network architectures
- Integration of pruning and quantization techniques
- Huffman encoding for additional compression
- Visualization of Pareto fronts
- Support for multiple datasets (CIFAR-10, ImageNet, etc.)

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- pymoo
- Matplotlib
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

1. Run the compression algorithm:
```bash
python reducer.py --arch resnet18 --dataset cifar10 --input_size 32
```

### Command Line Arguments

- `--data`: Location of the data corpus
- `--dataset`: Dataset to use (cifar10, imagenet, etc.)
- `--input_size`: Size of input images
- `--arch`: Model architecture
- `--num_workers`: Number of data loading workers
- `--num_classes`: Number of classes in the dataset

## Project Structure

- `reducer.py`: Main optimization algorithm implementation
- `merger.py`: Model merging utilities
- `dataset.py`: Dataset handling and preprocessing
- `utils.py`: Utility functions
- `compression/`: Compression techniques implementation
- `checkpoints/`: Checkpoints for pre-trained deep neural networks on CIFAR-10 and CIFAR-100 datasets.
- `models/`: Neural network model definitions
- `arg_parser.py`: Command line argument parsing

## License

This project is licensed under the terms of the included LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{moea-compression,
  author = {Rasa Khosrowshahli, Shahryar Rahnamayan, Beatrice Ombuki-Berman},
  title = {MOEA Compression: Multi-objective Neural Network Compression},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/moea-compression}
}
```
