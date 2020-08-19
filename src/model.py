from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import torch.nn as nn
import os
import torch
import wandb
import cv2


def xavier_init(layer):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

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
        self.encoder = nn.ModuleList(
                    [
                        nn.Conv1d(in_channels=1, out_channels=16,
                            kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0, return_indices=True),
                        nn.Conv1d(in_channels=16, out_channels=32,
                            kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0, return_indices=True),
                        nn.Conv1d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True)
                        ]
                        )
        self.decoder = nn.ModuleList([
                        nn.Conv1d(in_channels=64, out_channels=32,
                            kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxUnpool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=32, out_channels=16,
                            kernel_size=3, stride=1,  padding=1),
                        nn.BatchNorm1d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxUnpool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=16, out_channels=1,
                            kernel_size=3, stride=1, padding=1)
                            ]
                        )

        self.indices = []


    def encode(self, x):
        self.indices = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool1d):
                x, ind = layer(x)
                self.indices.append(ind)
            else:
                x = layer(x)

        return x

    def decode(self, x):
        index = len(self.indices) - 1
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool1d):
                x = layer(x, self.indices[index])
                index -= 1
            else:
                x = layer(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))



class Classifier(nn.Module):
    """
    The Classifier is a small neural netork that learns to classify
    bird audio waves based on the features that are extracted previosly.
    An aribrary encoder must be provide that is compatible with the
    classification layers.
    """
    def __init__(self, encoder, w_init=xavier_init):
        super().__init__()
        self.encoder = encoder
        self.total_labels = 264
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=64, out_channels=32,
                kernel_size=3, stride=1,  padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=16,
                kernel_size=3, stride=1,  padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=1,
                kernel_size=3, stride=1,  padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 264),
            nn.Softmax(dim=0),
        ])
        for layer in self.layers:
            w_init(layer)


    def forward(self, x):
        x = self.encoder.encode(x) # feature extraction
        for layer in self.layers:
            x = layer(x)
        return x.reshape(x.shape[0], -1)

def to_cpu(t):
    return t.cpu().detach().numpy()



def train_step_classification(model, data_loader, optimizer, loss_criterion, verbose_epochs, device, step):
    model.train()
    prediction_threshold = 0.9
    for i, data in enumerate(data_loader):
        print('[' + str(i) + '/' + len(data_loader) + ']')
        optimizer.zero_grad()
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        labels = labels.to(device)
        out = model(data)
        loss = loss_criterion(out, labels.long())
        loss.backward()
        optimizer.step()
        if i % verbose_epochs == 0:
            print('Train Loss:', loss.item())
            one_hot_labels = torch.zeros(out.shape)
            for i, label in enumerate(labels):
                one_hot_labels[i][label] = 1.0
            out = (out > prediction_threshold)
            one_hot_labels = to_cpu(one_hot_labels)
            one_hot_labels.max()
            out = to_cpu(out)
           #  mat = multilabel_confusion_matrix(one_hot_labels, out)
            a = accuracy_score(one_hot_labels, out)
            f1 = f1_score(one_hot_labels, out, average='micro', zero_division=0)
            p = precision_score(one_hot_labels, out, average='micro', zero_division=0)
            r = recall_score(one_hot_labels, out, average='micro', zero_division=0)
            print('Train accuracy:', a)
            print('F1 score:', f1)
            print('Train Precision score: ', p)
            print('Train Recall score: ', r)
            wandb.log({'Train Loss' : loss.item(), 'Train accuracy': a,
                        'Train F1  score' : f1, 'Train precision' : p,
                        'Train Recall score:' : r}, step=step)
            step += 1

        return step



def eval_step_classification(model, data_loader, loss_criterion, verbose_epochs, device, step):
    model.eval()
    prediction_threshold = 0.9
    for i, data in enumerate(data_loader):
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        labels = labels.to(device)
        out = model(data)
        loss = loss_criterion(out, labels.long())
        if i % verbose_epochs == 0:
            print('[' + str(i) + '/' + len(data_loader) + ']')
            print('Eval Loss:', loss.item())
            one_hot_labels = torch.zeros(out.shape)
            for i, label in enumerate(labels):
                one_hot_labels[i][label] = 1.0
            out = (out > prediction_threshold)
            one_hot_labels = to_cpu(one_hot_labels)
            one_hot_labels.max()
            out = to_cpu(out)
            #mat = multilabel_confusion_matrix(one_hot_labels, out)
            a = accuracy_score(one_hot_labels, out)
            f1 = f1_score(one_hot_labels, out, average='micro', zero_division=0)
            p = precision_score(one_hot_labels, out, average='micro', zero_division=0)
            r = recall_score(one_hot_labels, out, average='micro', zero_division=0)
            print('Eval accuracy:', a)
            print('F1 score:', f1)
            print('Eval Precision score: ', p)
            print('Eval Recall score: ', r)
            wandb.log({'Eval Loss' : loss.item(), 'Eval accuracy': a,
                        'Eval F1  score' : f1, 'Eval precision' : p,
                        'Eval Recall score:' : r}, step=step)
            step += 1
    return step


def train_step(model, data_loader, optimizer, loss_criterion, verbose_epochs, device, step):
    model.train()
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        out = model(data)
        loss = loss_criterion(out, data).sum() # BEWARE: overflow might happen since we are summing all losses
        loss.backward()
        optimizer.step()
        if i % verbose_epochs == 0:
            print('[' + str(i) + '/' + len(data_loader) + ']')
            print('Train Loss:', loss.item())
            wandb.log({'Train Loss': loss.item()}, step=step)
            step += 1
    return step



def eval_step(model, data_loader, loss_criterion, verbose_epochs, device, step):
    model.eval()
    for i, data in enumerate(data_loader):
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        out = model(data)
        loss = loss_criterion(out, data).sum()
        if i % verbose_epochs == 0:
            print('[' + str(i) + '/' + len(data_loader) + ']')
            print('Eval Loss:', loss.item())
            wandb.log({'Eval Loss' :  loss.item()}, step=step)
            step += 1
    return step
