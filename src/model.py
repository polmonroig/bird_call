import torch.nn as nn 
import torch 


class FeatureExtractor(nn.Module):

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

    def __init__(self, encoder):
        super().__init__() 
        self.encoder = encoder
