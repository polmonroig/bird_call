import torch.nn as nn 
import torch 


class FeatureExtractor(nn.Module):
    """
    The FeatureExtractor is an autoencoder model that works by learning 
    a mapping of the raw audio wave into a smaller dimensional space 
    with high level features. This model is train in an unsupervised manner 
    as an undercomplete autoencoder. The objective is to create a model 
    that extracts the most important features of the audio, similarly to 
    a log-mel audio transform but with a noise reduction. It must learn to 
    represent audio without the noise. 

    """
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=16, 
                            kernel_size=3, stride=1, padding=0), 
                        nn.Conv1d(in_channels=16, out_channels=32,
                            kernel_size=3, stride=1, padding=0), 
                        nn.Conv1d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1, padding=0)
                        )
        self.decoder = nn.Sequential(
                        nn.Conv1d(in_channels=64, out_channels=32,
                            kernel_size=3, stride=1, padding=0), 
                        nn.Conv1d(in_channels=32, out_channels=16,
                            kernel_size=3, stride=1,  padding=0), 
                        nn.Conv1d(in_channels=16, out_channels=1,
                            kernel_size=3, stride=1, padding=0)
                        )


    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))



class Classifier(nn.Module):
    """
    The Classifier is a small neural netork that learns to classify 
    bird audio waves based on the features that are extracted previosly. 
    An aribrary encoder must be provide that is compatible with the 
    classification layers. 
    """
    def __init__(self, encoder):
        super().__init__() 
        self.encoder = encoder
