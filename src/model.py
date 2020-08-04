import torch.nn as nn 
import torch 


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        
    def encode(self, x):
        return x 

    def decode(self, x):
        return x 

    def forward(self, x):
        return self.decode(self.encode(x))



class Classifier(nn.Module):

    def __init__(self, encoder):
        super().__init__() 
        self.encoder = encoder
