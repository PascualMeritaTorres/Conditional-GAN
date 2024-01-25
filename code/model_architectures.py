import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGeneratorModel(nn.Module):
    """
    Generator model for generating images.

    Attributes:
        num_classes (int): Number of classes in the dataset. (To determine embedding size).
        input_dim (int): Dimension of the input vector. (Noise + label embedding size).
        output_dim (int): Dimension of the output vector (Image size flattened).

    Methods:
        forward: Defines the forward pass.
    """

    def __init__(self, num_classes=10, input_dim=100+10, output_dim=784):
        super(ConditionalGeneratorModel, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        return self.main(x)


class ConditionalDiscriminatorModel(nn.Module):
    """
    Discriminator model for classifying images.

    Attributes:
        num_classes (int): Number of classes in the dataset. (To determine embedding size).
        input_dim (int): Dimension of the input vector. (Noise + label embedding size).
        output_dim (int): Dimension of the output vector (Image size flattened).

    Methods:
        forward: Defines the forward pass.
    """

    def __init__(self, num_classes=10, input_dim=784 + 10, output_dim=1):
        super(ConditionalDiscriminatorModel, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        return self.main(x)





class StandardGeneratorModel(nn.Module):
    """
    Generator model for generating images.

    Attributes:
        num_classes (int): Number of classes in the dataset. (To determine embedding size).
        input_dim (int): Dimension of the input vector. (Noise + label embedding size).
        output_dim (int): Dimension of the output vector (Image size flattened).

    Methods:
        forward: Defines the forward pass.
    """

    def __init__(self, input_dim=100, output_dim=784):
        super(StandardGeneratorModel, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class StandardDiscriminatorModel(nn.Module):
    """
    Discriminator model for classifying images.

    Attributes:
        num_classes (int): Number of classes in the dataset. (To determine embedding size).
        input_dim (int): Dimension of the input vector. (Noise + label embedding size).
        output_dim (int): Dimension of the output vector (Image size flattened).

    Methods:
        forward: Defines the forward pass.
    """

    def __init__(self, input_dim=784, output_dim=1):
        super(StandardDiscriminatorModel, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

