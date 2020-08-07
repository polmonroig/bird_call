from model import FeatureExtractor, train_step, eval_step
from utils import filesystem
import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train the neural network')

    return parser


def wandb_parameter_register(args):
    pass

def train_autoencoder(device, args):
    # model definition
    model = FeatureExtractor()
    model.to(device)
    # data definition
    train_chunks =
    train_dataset = GenerativeDataset()

    # main loop
    optimizer = optim.SGD()
    for epoch in args.n_epochs:
        print('Epoch:', epoch, '/', args.n_epochs)


    print(model)

def train_classifier(device, args):
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

    wandb_parameter_register(args)
    train_autoencoder(device, args)
    #train_classifier(device, args)

if __name__ == '__main__':
    main()
