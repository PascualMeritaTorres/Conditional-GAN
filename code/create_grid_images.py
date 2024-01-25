import torch
from torchvision import transforms
from model_architectures import *
from storage_utils import load_from_stats_pkl_file
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def load_generator(model_path, device, gan_type='cgan'):
    # Initialize the GeneratorModel with any required parameters
    if gan_type=='gan':
         model = StandardGeneratorModel()
    else:
        model = ConditionalGeneratorModel() 

    checkpoint = torch.load(model_path, map_location=device)
    generator_state_dict = {k[len("generator."):]: v for k, v in checkpoint['network'].items() if k.startswith("generator.")}
    model.load_state_dict(generator_state_dict)
    model.to(device)
    model.eval()
    return model

def generate_images(generator, digit, device, gan_type='cgan'):
    """
    Generates images using the generator model.
    For CGAN, a digit label is used. For standard GAN, no label is required.
    """
    images = []
    for _ in range(2):  # Generate two images
        noise = torch.randn(1, 100).to(device)
        if gan_type == 'cgan':
            label = torch.LongTensor([digit]).to(device)
            with torch.no_grad():
                generated_image = generator(noise, label)
        else:
            with torch.no_grad():
                generated_image = generator(noise)
        generated_image = generated_image.cpu().view(1, 28, 28)
        images.append(generated_image)
    return images

def save_grid_image(images, filename):
    """
    Arranges images in a grid and saves them as a single image.
    """
    num_rows = 2
    num_cols = 10 if images[0][0].shape[0] != 1 else len(images)  # 10 for CGAN, variable for GAN
    grid = Image.new('L', (28 * num_cols, 28 * num_rows))

    for i, img_set in enumerate(images):
        for j, img in enumerate(img_set):
            img = transforms.ToPILImage()(img.squeeze(0))
            grid.paste(img, (i * 28, j * 28))

    directory = os.path.dirname(filename)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    grid.save(filename)

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_grid_images.py <model's epoch to use> <gan_type>")
        sys.exit(1)

    model_epoch = int(sys.argv[1])
    gan_type = sys.argv[2]

    if model_epoch < 0:
        print("Model Epoch must be larger than 0.")
        sys.exit(1)

    if gan_type == 'cgan':
        model_path = f'../Cgan_experiment/saved_models/train_model_{model_epoch}'
    elif gan_type == 'gan':
        model_path = f'../gan_experiment/saved_models/train_model_{model_epoch}'
    else:
        print("Please input a valid gan type")
        sys.exit(1)


    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.device_count() > 0:
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    generator = load_generator(model_path, device, gan_type)

    all_images = []
    if gan_type == 'cgan':
        for digit in range(10):
            generated_images = generate_images(generator, digit, device, gan_type)
            all_images.append(generated_images)
    else:
        # For GAN, generate a set number of images without labels
        for _ in range(10):  # Arbitrary number of images
            generated_images = generate_images(generator, None, device, gan_type)
            all_images.append(generated_images)

    filename = 'generated_images_grid.png'
    save_grid_image(all_images, filename)
    print(f'Generated grid image saved as {filename}')

if __name__ == '__main__':
    main()
