from model import FeatureExtractor 
import torch.optim as optim 
import torch.nn as nn 
import torch 
import wandb
import argparse 

def get_parser():
    parser = argparse.ArgumentParser(description='Train the neural network')

    return parser 


def train_autoencoder(device):
    model = FeatureExtractor() 
    model.to(device)
    print(model) 

def train_classifier(device):
    raise NotImplementedError()

def main():
    print('Initializing model training...')
    parser = get_parser() 
    args = parser.parse_args() 
    device = torch.device('cpu')
    if torch.cuda.is_available(): 
        print('Running on gpu')
        device = torch.device('cuda')
    else:
        print('Running on cpu')

    train_autoencoder(device)
    #train_classifier(device)

if __name__ == '__main__':
    main()
