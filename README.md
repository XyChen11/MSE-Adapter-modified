# MM-Adapter-DDPversion

This project is a modified version of the [MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter.git) framework, designed for multi-modal learning tasks. The main enhancements include support for Distributed Data Parallel (DDP) multi-GPU training and improvements to several modules for better performance and flexibility.

## Features

- **DDP Multi-GPU Training:** Efficient distributed training using PyTorch's DDP, enabling scalable experiments across multiple GPUs.
- **Module Improvements:** Refactored and optimized modules for classification and regression tasks.
- **Flexible Configuration:** Easy-to-use configuration files for different tasks and experiments.
- **Data Preprocessing:** Comprehensive scripts for feature extraction, text preprocessing, and data loading.

## Directory Structure

- config: Configuration files for classification and regression.
- data: Data preprocessing and feature extraction scripts.
- models: Model definitions, including custom and ChatGLM3 models.
- trains: Training scripts for various models.
- utils: Utility functions and metrics.
- logs, results: Logging and result storage.

## Getting Started

1. **Install Dependencies**

   - Python 3.8+
   - PyTorch (with CUDA support)
   - Other required packages (see requirements.txt or install as needed)
     ```
     git clone https://github.com/Chandler-Chen11/MSE-Adapter-modified.git
     cd MSE-Adapter-modified
     conda create --name MSE-Adapter-modified python=3.10.13
     conda activate MSE-Adapter-modified
     pip install -r requirements.txt
     # or conda env create -f environment.yml
     ```
2. **Prepare Data**

   - Place your datasets in the data directory.
   - Use provided scripts for preprocessing.
3. **Configure Training**

   - Edit configuration files in config for your task.
   - Edit the augments in run.py file to change some hyperparameters.
4. **Run Training**

   - For DDP multi-GPU training:
     ```
     CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --use_ddp
     ```
   - For single GPU/CPU training:
     ```
     python run.py
     ```

## Citation

If you use this project, please cite the original MSE-Adapter paper.

## License

This project is licensed under the terms of the LICENSE file.
