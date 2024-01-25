import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from gan_interview.arg_extractor import get_args
from gan_interview.experiment_builder import ExperimentBuilder
from gan_interview.model_architectures import *
import os 
from matplotlib import pyplot as plt



if __name__ == "__main__":

    args = get_args()  # get arguments from command line
    rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    torch.manual_seed(seed=args.seed)  # sets pytorch's seed


    if args.dataset == 'MNIST':
        # set up data augmentation transforms for training 
        transform_data = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ]) 
        all_train_data = datasets.MNIST('data', train=True, download=True, transform=transform_data)
        train_data_loader= DataLoader(all_train_data, batch_size=args.batch_size, shuffle=True)
        num_channels=1
        num_classes=10
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported. Please choose a valid dataset.")

    test_data = datasets.MNIST('data', train=False, download=True, transform=transform_data)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    if args.gan_type == 'cgan':
        # generator_model=GeneratorModel(image_size,num_channels,num_classes)
        # discriminator_model=DiscriminatorModel(image_size,num_channels,num_classes)
        # Instantiate models and move them to the device
        discriminator_model = ConditionalDiscriminatorModel()
        generator_model = ConditionalGeneratorModel()
    elif args.gan_type == 'gan':
        # Instantiate standard GAN models
        discriminator_model = StandardDiscriminatorModel()
        generator_model = StandardGeneratorModel()
    else:
        raise ValueError("Invalid model type specified.")

    # build an experiment object
    gan_experiment = ExperimentBuilder(generator=generator_model, 
                                        discriminator=discriminator_model,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=args.continue_from_epoch,
                                        train_data=train_data_loader,
                                        gan_type=args.gan_type,
                                        test_data=test_data_loader,
                                        learning_rate=args.learning_rate,
                                        batch_size=args.batch_size)  
    

    experiment_metrics, test_metrics = gan_experiment.run_experiment()  # run experiment and return experiment metrics
