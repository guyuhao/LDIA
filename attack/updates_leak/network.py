"""
Attack model architecture for Updates-Leak
"""
import torch.nn as nn


# Model for the multi-sample label estimation attack for the MNIST dataset
class labelPredMNIST(nn.Module):
    def __init__(self, attackInput, numOfClasses):
        super(labelPredMNIST, self).__init__()
        # The encoder is considered all layers except the last one, which is considered to be the decoder in this attack.
        self.preAttack = nn.Sequential(
            nn.Linear(attackInput, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, numOfClasses),   
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inputAttack):
        return self.preAttack(inputAttack)
