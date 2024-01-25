import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from gan_interview.storage_utils import save_statistics
from model_architectures import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
from torchvision import transforms

class ExperimentBuilder(nn.Module):
    def __init__(self, generator, discriminator, experiment_name, num_epochs, train_data, gan_type,
                 test_data, use_gpu, continue_from_epoch=-1, learning_rate=1e-3, batch_size=100):
        """
        This object takes care of running training and evaluation and saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param generator: A pytorch nn.Module which implements a GAN's Generator
        :param discriminator: A pytorch nn.Module which implements a GAN's Discriminator
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param gan_type: Type of architecture used.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param batch_size: Batch_size for experiment.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()


        self.experiment_name = experiment_name
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.continue_from_epoch=continue_from_epoch
        self.num_epochs = num_epochs
        self.gan_type=gan_type

        if torch.backends.mps.is_available() and use_gpu:
            self.device = torch.device("mps")
            self.generator.to(self.device)
            self.discriminator.to(self.device)
            print('Use mps', self.device)
        elif torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.generator.to(self.device)
            self.discriminator.to(self.device)
            self.generator = nn.DataParallel(module=self.generator)
            self.discriminator = nn.DataParallel(module=self.discriminator)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device =  torch.cuda.current_device()
            self.generator.to(self.device)
            self.discriminator.to(self.device)
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)


        self.train_data = train_data
        self.test_data = test_data

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.loss = nn.BCELoss()
        
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))


        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.loss = nn.BCELoss().to(self.device)  # send the loss computation to the GPU

        if self.continue_from_epoch == -2:  # if continue from epoch is -2 then continue from latest saved model
            self.state= self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')
            self.starting_epoch = int(self.state['model_epoch'])

        elif self.continue_from_epoch > -1:  # if continue from epoch is greater than -1 then
            self.state= self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=self.continue_from_epoch)
            self.starting_epoch = self.continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0

 
    def save_model(self, model_save_dir, model_save_name, model_idx):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        self.state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(self.state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"g_train_loss": [], "d_train_loss": []}
        with tqdm.tqdm(total=self.num_epochs) as pbar_train:  # create a progress bar for training
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                current_epoch_losses = {"g_train_loss": [], "d_train_loss": []}

                for batch_index, data in enumerate(self.train_data):
                    if self.gan_type == 'gan':
                        # Generating noise for the generator
                        noise = torch.randn(self.batch_size, 100).to(self.device)
                        generated_images = self.generator(noise) # Creates fake images

                        # Preparing real data and labels for the discriminator
                        real_images = data[0].view(self.batch_size, 784).to(self.device)  # Flatten the images
                        real_labels = torch.ones(self.batch_size).to(self.device)  # Labels for real data

                        # Zero gradients for discriminator
                        self.discriminator_optimizer.zero_grad()

                        # Discriminator loss on real data
                        discriminator_real_output = self.discriminator(real_images).view(self.batch_size)
                        loss_on_real = self.loss(discriminator_real_output, real_labels)

                        # Discriminator loss on generated (fake) data
                        discriminator_fake_output = self.discriminator(generated_images.detach()).view(self.batch_size)
                        loss_on_fake = self.loss(discriminator_fake_output, torch.zeros(self.batch_size).to(self.device))
                                            
                        # Total discriminator loss
                        total_discriminator_loss = (loss_on_real + loss_on_fake) / 2
                        total_discriminator_loss.backward()
                        self.discriminator_optimizer.step()

                        current_epoch_losses["d_train_loss"].append(total_discriminator_loss.item())

                        # Training Generator
                        self.generator_optimizer.zero_grad()

                        # Generate fake images again for the generator update
                        generated_images_for_update = self.generator(noise)
                        discriminator_output_for_generator = self.discriminator(generated_images_for_update).view(self.batch_size)

                        # Generator loss
                        generator_loss = self.loss(discriminator_output_for_generator, real_labels) # Take into account real_labels are all 1's
                        generator_loss.backward()
                        self.generator_optimizer.step()

                        current_epoch_losses["g_train_loss"].append(generator_loss.item())

                    else:
                        # Generating noise and fake labels for the generator
                        noise = torch.randn(self.batch_size, 100).to(self.device)
                        fake_labels = torch.randint(0, 10, (self.batch_size,)).to(self.device)
                        generated_images = self.generator(noise, fake_labels) # Creates fake images

                        # Preparing real data and labels for the discriminator
                        real_images = data[0].view(self.batch_size, 784).to(self.device)  # Flatten the images
                        real_labels = torch.ones(self.batch_size).to(self.device)  # Labels for real data

                        # Zero gradients for discriminator
                        self.discriminator_optimizer.zero_grad()

                        # Discriminator loss on real data
                        discriminator_real_output = self.discriminator(real_images, data[1].to(self.device)).view(self.batch_size)
                        loss_on_real = self.loss(discriminator_real_output, real_labels)

                        # Discriminator loss on generated (fake) data
                        discriminator_fake_output = self.discriminator(generated_images.detach(), fake_labels).view(self.batch_size)
                        loss_on_fake = self.loss(discriminator_fake_output, torch.zeros(self.batch_size).to(self.device))
                        
                        # Total discriminator loss
                        total_discriminator_loss = (loss_on_real + loss_on_fake) / 2
                        total_discriminator_loss.backward()
                        self.discriminator_optimizer.step()

                        current_epoch_losses["d_train_loss"].append(total_discriminator_loss.item())
                        
                        # Training Generator
                        self.generator_optimizer.zero_grad()

                        # Generate fake images again for the generator update
                        generated_images_for_update = self.generator(noise, fake_labels)
                        discriminator_output_for_generator = self.discriminator(generated_images_for_update, fake_labels).view(self.batch_size)

                        # Generator loss
                        generator_loss = self.loss(discriminator_output_for_generator, real_labels) # Take into account real_labels are all 1's
                        generator_loss.backward()
                        self.generator_optimizer.step()
                        
                        current_epoch_losses["g_train_loss"].append(generator_loss.item())

                pbar_train.update(1)
                pbar_train.set_description(' g_train_loss: %.3f, d_train_loss: %.3f' % (
                       torch.mean(torch.FloatTensor(current_epoch_losses["g_train_loss"])), torch.mean(torch.FloatTensor(current_epoch_losses["d_train_loss"])))
                )
                print()

                # Aggregate and calculate the mean of all metrics for the current epoch.
                for key, value in current_epoch_losses.items():
                    total_losses[key].append(torch.mean(torch.FloatTensor(value)).item())

                # Save the statistics to a summary CSV file.
                save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                                stats_dict=total_losses, current_epoch=epoch,
                                continue_from_mode=True if (self.starting_epoch != 0 or epoch > 0) else False)
                
                # Update the state and save the model.
                self.state['model_epoch'] = epoch
                self.save_model(model_save_dir=self.experiment_saved_models,
                                model_save_name="train_model", model_idx=epoch,)

        

        # Test set evaluation.
        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models,
                        model_save_name="train_model", 
                        model_idx=self.num_epochs-1)
        current_epoch_losses = {"g_test_loss": [], "d_test_loss": []}
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:
            for input_data, target_labels in self.test_data:
                self.generator.eval()
                self.discriminator.eval()

                noise = torch.randn(input_data.size(0), 100).to(self.device)
                real_images = input_data.view(input_data.size(0), 784).to(self.device)
                real_labels = torch.ones(input_data.size(0)).to(self.device)
                fake_labels_for_discriminator = torch.zeros(input_data.size(0)).to(self.device)


                if self.gan_type == 'gan':
                    # Logic for standard GAN
                    generated_images = self.generator(noise)

                else:
                    # Logic for CGANs
                    fake_labels = torch.randint(0, 10, (input_data.size(0),)).to(self.device)
                    generated_images = self.generator(noise, fake_labels)


                with torch.no_grad():
                    # Discriminator loss on generated (fake) data
                    if self.gan_type == 'gan':
                        # For standard GAN, the discriminator doesn't use label information for fake data
                        discriminator_real_output = self.discriminator(real_images).view(self.batch_size)
                        discriminator_fake_output = self.discriminator(generated_images).view(input_data.size(0))
                    else:
                        # For CGANs, the discriminator uses label information for fake data
                        discriminator_real_output = self.discriminator(real_images, target_labels.to(self.device)).view(input_data.size(0))
                        discriminator_fake_output = self.discriminator(generated_images, fake_labels).view(input_data.size(0))

                    loss_on_real = self.loss(discriminator_real_output, real_labels)     
                    loss_on_fake = self.loss(discriminator_fake_output, fake_labels_for_discriminator)
                    
                    # Total discriminator loss
                    total_discriminator_loss = (loss_on_real + loss_on_fake) / 2

                    # Generator loss
                    if self.gan_type == 'gan':
                        # For standard GAN
                        discriminator_output_for_generator = self.discriminator(generated_images).view(input_data.size(0))
                    else:
                        # For CGANs
                        discriminator_output_for_generator = self.discriminator(generated_images, fake_labels).view(input_data.size(0))

                    generator_loss = self.loss(discriminator_output_for_generator, real_labels)

                    current_epoch_losses["g_test_loss"].append(generator_loss.item())
                    current_epoch_losses["d_test_loss"].append(total_discriminator_loss.item())

                    pbar_test.update(1)


        # Save test set metrics.
        test_losses = {key: [torch.mean(torch.FloatTensor(value)).item()] for key, value in current_epoch_losses.items()}
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses
