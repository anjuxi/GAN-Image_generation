# GAN Image Generator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15%2B-green)](https://pytorch.org/vision/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-yellow)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-red)](https://matplotlib.org/)
[![tqdm](https://img.shields.io/badge/tqdm-4.65%2B-purple)](https://github.com/tqdm/tqdm)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

This project implements a standard generative adversarial network (GAN) for generating handwritten digit images. A GAN has two networks:

- **Generator** — takes a random noise vector and outputs a fake image
- **Discriminator** — decides whether an input image is real or generated

Through adversarial training, the generator gradually learns to produce handwritten digits that look real.

MNIST is used here for demonstration. Other datasets can be used in practice.

## Usage

### Option 1: Jupyter Notebook

```bash
jupyter notebook gan-mnist.ipynb
```

Run the cells in order.

## Main Features

### 1. Data Handling

- Downloads MNIST automatically
- Normalizes and standardizes the data
- Uses an efficient data loader (multi-process and GPU acceleration supported)

### 2. Model Architecture

#### Generator

- Input: 100-dimensional random noise vector
- Output: 28×28 grayscale image
- Structure: fully connected layers + transposed convolution layers

#### Discriminator

- Input: 28×28 grayscale image
- Output: real/fake probability (0–1)
- Structure: convolution layers + fully connected layers

### 3. Training

- ✅ Single-GPU and multi-GPU training (detected automatically)
- ✅ Resume from a checkpoint (`continue_train = True`)
- ✅ Dynamic learning-rate adjustment
- ✅ Training progress with tqdm
- ✅ Periodic model and sample saving

### 4. Evaluation

- Generated sample grid visualization
- Latent-space interpolation

## Hyperparameters

| Parameter | Default | Description |
| ------------ | ------ | ------------ |
| `batch_size` | 256 | Batch size |
| `epochs` | 25 | Training epochs |
| `lr_g` | 0.0002 | Generator learning rate |
| `lr_d` | 0.0002 | Discriminator learning rate |
| `latent_dim` | 100 | Noise vector dimension |
| `img_size` | 28 | Image size |
| `channels` | 1 | Number of image channels |

## GPU Support

This project is tuned for 2× T4 GPUs:

- `nn.DataParallel` for multi-GPU training
- A larger batch size to use GPU memory
- `pin_memory` to speed up data transfer
- Asynchronous data loading

## Resume Training

To continue from an existing model, change:

```python
continue_train = True
continue_model_path = "./models/generator_epoch_20.pth"
```

## Generated Samples

After training, samples are saved under `./samples/` (uncomment the corresponding code):

- `real_samples_preview.png` — preview of real data
- `epoch_*_batch_*.png` — samples generated during training
- `generated_grid_8x8.png` — 8×8 sample grid
- `interpolation_samples.png` — latent-space interpolation

## License

[MIT License](LICENSE)

## Author

**Dr. Ailan Anjuxi (谙弆悕)**

- Email: [anjuxi.ME@outlook.com](mailto:anjuxi.ME@outlook.com)
- SIP: [sip:anjuxi@sip.linphone.org](sip:anjuxi@sip.linphone.org)
