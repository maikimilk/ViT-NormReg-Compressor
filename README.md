# Norm-Regularized Token Compression in Vision Transformer Networks

![AI](https://img.shields.io/badge/AI-Image_Classification-blue)
![Pruning](https://img.shields.io/badge/Pruning-Vision_Transformer-red)
![Python](https://img.shields.io/badge/code-Python-green)

## Overview
This repository contains the official implementation of the paper "Norm-Regularized Token Compression in Vision Transformer Networks". The paper proposes a novel method for token compression in Vision Transformer networks using norm regularization to enhance model efficiency and performance.

## Prerequisites
This project requires the following libraries:
- PyTorch
- torchvision
- timm
- numpy
- tqdm
- thop (for calculating the model complexity)

Ensure you have Python 3.x installed along with the above libraries.

## Installation
Clone this repository to your local machine to get started:
```bash
git clone https://github.com/yourgithubusername/ViT-NormReg-Compressor.git
cd ViT-NormReg-Compressor
```

## Installation
To use the model, modify the inputs in main.py. You can change the model argument to apply different token compression techniques. Adjust the batch_size argument in the same file to set the desired batch size. To switch datasets (e.g., STL10, CIFAR10), modify the data_name argument.

## Code Structure
The project is structured as follows:

- main.py: Main script where models are configured and training is initiated.
- pruning/patch/timm: This directory contains implementations of the pruning methods we have applied to Vision Transformer models using the TIMM library.
- data/: Dataset handling scripts.

## Experimental Results
Applying norm regularization with the Top K method has shown to improve accuracy significantly in our experiments.

## Authors and Contributors
GitHub Username: maikimilk

## Acknowledgments

This project is based in part on the code and concepts from the following research:

- Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. "Token Merging: Your ViT but Faster." In International Conference on Learning Representations, 2023.

This project also makes use of third-party data:

- "Data Set Title" by Creator Name, available under a Creative Commons Attribution-NonCommercial [CC-BY-NC 4.0](LICENSE). [View Source](https://github.com/facebookresearch/ToMe)


## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{masayuki2024norm-pruning,
  title={Norm-Regularized Token Compression in Vision Transformer Networks},
  author={Masayuki Ishikawa},
  year={2024}
}

@inproceedings{bolya2022tome,
  title={Token Merging: Your {ViT} but Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the [CC-BY-NC 4.0](LICENSE) file for details.

## Contact
For questions and feedback, please reach out to ri0146fe@ed.ritsumei.ac.jp