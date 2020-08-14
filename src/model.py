from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torch
import wandb


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
            nn.Conv1d(in_channels=64, out_channels=1,
                kernel_size=3, stride=1, padding=1),
            nn.AdaptiveMaxPool1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 264),
            nn.Softmax(),
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



def train_step_classification(model, data_loader, optimizer, loss_criterion, verbose_epochs, device):
    model.train()
    prediction_threshold = 0.5
    for i, data in enumerate(data_loader):
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
            print(labels.shape)
            print(out.shape)
            print(one_hot_labels.shape)
            for sample, label in zip(one_hot_labels, labels):
                sample = label
            out = (out > prediction_threshold)
            one_hot_labels = to_cpu(one_hot_labels)
            out = to_cpu(out)
            a = accuracy_score(one_hot_labels, out)
            f1 = f1_score(one_hot_labels, out, average='micro', zero_division=0)
            print('Train accuracy:', a)
            print('F1 score:', f1)
            wandb.log({'Train Loss' : loss.item(), 'Train accuracy': a, 'F1  score' : f1})



def eval_step_classification(model, data_loader, loss_criterion, verbose_epochs, device):
    model.eval()
    for i, data in enumerate(data_loader):
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        labels = labels.to(device)
        out = model(data)
        loss = loss_criterion(out, labels.long())
        if i % verbose_epochs == 0:
            print('Eval Loss:', loss.item())
            one_hot_labels = torch.zeros(out.shape)
            print(labels.shape)
            print(out.shape)
            print(one_hot_labels.shape)
            for sample, label in zip(one_hot_labels, labels):
                sample = label
            out = (out > prediction_threshold)
            one_hot_labels = to_cpu(one_hot_labels)
            out = to_cpu(out)
            a = accuracy_score(one_hot_labels, out)
            f1 = f1_score(one_hot_labels, out, average='micro', zero_division=0)
            print('Eval accuracy:', a)
            print('Eval F1 score:', f1)
            wandb.log({'Eval Loss' : loss.item(), 'Eval accuracy': a, 'Eval F1  score' : f1})


def train_step(model, data_loader, optimizer, loss_criterion, verbose_epochs, device):
    model.train()
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        out = model(data)
        loss = loss_criterion(out, data)
        loss.backward()
        optimizer.step()
        if i % verbose_epochs == 0:
            print('Train Loss:', loss.item())
            wandb.log({'Train Loss:', loss.item()})



def eval_step(model, data_loader, loss_criterion, verbose_epochs, device):
    model.eval()
    for i, data in enumerate(data_loader):
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        out = model(data)
        loss = loss_criterion(out, data)
        if i % verbose_epochs == 0:
            print('Eval Loss:', loss.item())
            wandb.log({'Eval Loss:', loss.item()})
