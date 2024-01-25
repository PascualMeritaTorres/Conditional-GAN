import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--dataset', nargs="?", type=str, default='MNIST', help='Name of Dataset to use')
    parser.add_argument('--image_size', nargs="?", type=int, default=28, help='Size of handled images')
    parser.add_argument('--gan_type', nargs="?", type=str, default='cgan', choices=['cgan', 'gan'], help='Type of GAN to use (cgan or gan)')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Epoch you want to continue training from while restarting an experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018, help='Seed to use for random number generator for experiment')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='Total number of epochs for model training')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1", help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True, help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--learning_rate', type=float, default='1e-3', help='Learning rate used in our optimizer')

    args = parser.parse_args()
    print(args)
    return args

