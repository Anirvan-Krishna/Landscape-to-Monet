# A CycleGAN Approach to Landscape-to-Painting Conversion

This repository contains the code implementation of a Cycle GAN model to convert landscape images to Monet-style paintings. The Cycle GAN architecture consists of generators and discriminators to achieve image-to-image translation.

## Overview

The project involves training two generators and discriminators:
- **Generators**: Converts between landscape images and Monet-style paintings.
- **Discriminators**: Distinguishes between real and generated images for both landscapes and paintings.

## Requirements

- Python 3.x
- PyTorch
- Albumentations

## Project Structure

The repository structure is organized as follows:
- `generator_model.py`: Contains the generator architecture.
- `discriminator_model.py`: Holds the discriminator architecture.
- `dataset.py`: Defines the dataset class for loading landscape and painting images.
- `utils.py`: Consists of utility functions for model checkpointing.
- `config.py`: Stores configuration parameters and transforms for data preprocessing.
- `train.py`: Main script to train the Cycle GAN model.

## Usage

### Clone the repository:
```bash
git clone https://github.com/your_username/landscape-to-monet.git
cd landscape-to-monet
```
### Configurations
Adjust the configuration parameters in `config.py` for specific learning rates, batch sizes, and other settings.

### Checkpoints
The trained generator models (`genp.pth.tar` and `genl.pth.tar`) and discriminator models (`criticl.pth.tar` and `criticp.pth.tar`) are saved periodically during training.

## Architecture
![CycleGAN Architecture](https://miro.medium.com/v2/resize:fit:1400/1*_KxtJIVtZjVaxxl-Yl1vJg.png)

### Key Features:
- **Unpaired Image Translation:** Enables translation between two domains without one-to-one correspondence in the training data.
- **Cycle Consistency:** Introduces cycle consistency loss to maintain image content during translation, allowing reconstruction back to the original domain.

### Workflow:
1. **Generator-Discriminator Pairs:** Comprises two generators and two discriminators, facilitating the mapping between the source and target domains while maintaining realism.
2. **Adversarial Training:** Involves adversarial training where the generators attempt to fool the discriminators, and the discriminators aim to differentiate between real and generated images.
3. **Cycle Consistency Loss:** Ensures that an image translated from domain A to domain B and then back to domain A remains similar to the original image in domain A.

Cycle GAN has found extensive use in various applications, including style transfer, image-to-image translation, and artistic image synthesis.

## Results
  | Input | Output |  
  |---------|---------|
  | ![Imgur](https://i.imgur.com/JCpH2Z9.jpg) |![Imgur](https://i.imgur.com/HbUkGVL.png) |

This is a result of 15 epochs of training for CycleGAN. Model shows significant improvement as it is trained further.

## Acknowledgements
For more details of the architecture, refer to: Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593). In Proceedings of the IEEE International Conference on Computer Vision (ICCV).


