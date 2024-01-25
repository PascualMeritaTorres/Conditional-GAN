
import torch
from torchvision import transforms
from model_architectures import *
from storage_utils import load_from_stats_pkl_file
import sys
import os
import torch
from torchvision import transforms
import sys
import os
from matplotlib import pyplot as plt
import matplotlib

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

def generate_image(generator, digit, device, gan_type='cgan'):
    # Generating noise for the generator
    noise = torch.randn(1, 100).to(device)

    # Conditional GAN requires a digit label
    if gan_type == 'cgan' and digit is not None:
        label = torch.LongTensor([digit]).to(device)
        with torch.no_grad():
            generated_image = generator(noise, label)
    else:
        with torch.no_grad():
            generated_image = generator(noise)

    generated_image = generated_image.cpu().view(1, 28, 28)
    return generated_image

def save_image(image, filename):
    """ Saves the generated image to the specified filename. """
    directory = os.path.dirname(filename)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    transforms.ToPILImage()(image.squeeze(0)).save(filename)

def main():
    # Check for minimum
    if len(sys.argv) < 3:
        print("Usage: python get_gan_output.py <model's epoch to use> <gan_type> [digit for cgan]")
        sys.exit(1)

    model_epoch = int(sys.argv[1])
    gan_type = sys.argv[2]
    digit = None  # Default

    # Parse digit if GAN type is cGAN
    if gan_type == 'cgan':
        if len(sys.argv) != 4:
            print("Digit must be provided for cGAN.")
            sys.exit(1)
        digit = int(sys.argv[3])

        if digit < 0 or digit > 9:
            print("Digit must be between 0 and 9 for cGAN.")
            sys.exit(1)

    if model_epoch<0:
        print("Model Epoch must be larger than 0.")
        sys.exit(1)

    # Adjusting model path based on gan_type
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
        print('Use mps', device)
    elif torch.cuda.device_count() > 1:
        device = torch.cuda.current_device()
        print('Use Multi GPU', device)
    elif torch.cuda.device_count() == 1:
        device =  torch.cuda.current_device()
        print('Use GPU', device)
    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
        print(device)

    generator = load_generator(model_path, device, gan_type)  # pass gan_type

    # Generate and save the image
    generated_image = generate_image(generator, digit, device, gan_type) 
    filename = f'image_output_{gan_type}_{digit if digit is not None else "NA"}.png'
    save_image(generated_image, filename)
    print(f'Generated image saved as {filename}')

if __name__ == '__main__':
    main()



