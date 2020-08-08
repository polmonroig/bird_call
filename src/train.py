from model import FeatureExtractor, train_step, eval_step
from utils import filesystem
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train the neural network')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs for the training loop')
    parser.add_argument('--batch_size', type=int, help='Batch size of the training and eval set')
    parser.add_argument('--verbose_epochs', type=int, help='Number of epochs per training verbose output')
    parser.add_argument('--lr', type=float, help='Learning rate of the optimizer')
    parser.add_argument('--eval_size', type=float, help='Evaluation dataset percentage')
    return parser


def wandb_parameter_register(args):
    pass

def train_autoencoder(device, args):
    # model definition
    model = FeatureExtractor()
    model.to(device)
    # data definition
    all_chunks = []
    # concatenate all chunk files
    # note that it is independent of the
    # class of each chunk sinc we are creating
    # a generative dataset
    for label in filesystem.listdir_complete(filesystem.train_audio_chunks_dir):
        chunks = filesystem.listdir_complete(label)
        all_chunks = all_chunks + chunks
    train_chunks, eval_chunks = train_test_split(all_chunks, test_size=args.eval_size)
    train_dataset = GenerativeDataset()

    # main loop
    optimizer = optim.SGD()
    for epoch in args.n_epochs:
        print('Epoch:', epoch, '/', args.n_epochs)
        train_step(model, )
        eval_step(model)

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
