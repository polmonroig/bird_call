from model import FeatureExtractor, train_step, train_step_classification, eval_step, eval_step_classification, Classifier
from data import GenerativeDataset, DiscriminativeDataset, LabelsEncoder
from utils import filesystem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from utils.soft_dtw import SoftDTW
from torchvision import transforms
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Train the neural network')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs for the training loop')
    parser.add_argument('--batch_size', type=int, help='Batch size of the training and eval set')
    parser.add_argument('--verbose_epochs', type=int, help='Number of epochs per training verbose output')
    parser.add_argument('--lr', type=float, help='Learning rate of the optimizer')
    parser.add_argument('--eval_size', type=float, help='Evaluation dataset percentage')
    parser.add_argument('--encoder_path', type=str, help='Path to feature extractor model state dict')
    return parser


def wandb_parameter_register(args):
    wandb.init(project='bird_call')
    wandb.config.n_epochs = args.n_epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.learning_rate = args.lr

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
    # transforms and dataset
    trf = normalize

    train_dataset = GenerativeDataset(train_chunks, transforms=trf)
    eval_dataset = GenerativeDataset(eval_chunks, transforms=trf)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, collate_fn=None,pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, collate_fn=None,pin_memory=True)

    # main loop
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_criterion = SoftDTW(gamma=1.0)
    for epoch in range(args.n_epochs):
        print('Epoch:', epoch, '/', args.n_epochs)
        train_step(model, train_dataloader, optimizer, loss_criterion, args.verbose_epochs, device)
        eval_step(model, eval_dataloader, loss_criterion, args.verbose_epochs, device)
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_checkpoint.pt'))

def train_classifier(device, args):
    encoder = FeatureExtractor()
    encoder.load_state_dict(torch.load(args.encoder_path))
    encoder.eval()
    classifier = Classifier(encoder)
    classifier.to(device)
    all_chunks = []
    all_labels = []
    for label in filesystem.listdir_complete(filesystem.train_audio_chunks_dir):
        chunks = filesystem.listdir_complete(label)
        all_chunks = all_chunks + chunks
        all_labels = all_labels + [label.split('/')[-1]] * len(chunks)
    train_chunks, eval_chunks, train_labels, eval_labels = train_test_split(all_chunks, all_labels, test_size=args.eval_size)

    # transforms and dataset
    trf = normalize
    # dataset generation
    labels_encoder = LabelsEncoder(pd.read_csv(filesystem.labels_encoding_file))
    train_dataset = DiscriminativeDataset(train_chunks, train_labels, labels_encoder, transforms=trf)
    eval_dataset = DiscriminativeDataset(eval_chunks, eval_labels, labels_encoder, transforms=trf)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, collate_fn=None,pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, collate_fn=None,pin_memory=True)

    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    loss_criterion = nn.CrossEntropyLoss()
    train_step = 0
    eval_step = 0
    for epoch in range(args.n_epochs):
        print('Epoch:', epoch, '/', args.n_epochs)
        train_step = train_step_classification(classifier, train_dataloader, optimizer, loss_criterion, args.verbose_epochs, device, train_step)
        eval_step = eval_step_classification(classifier, eval_dataloader, loss_criterion, args.verbose_epochs, device, eval_step)
        torch.save(classifier.state_dict(), os.path.join(wandb.run.dir, 'model_checkpoint.pt'))


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
