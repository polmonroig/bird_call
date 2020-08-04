from model import FeatureExtractor 
import torch.optim as optim 
import torch.nn as nn 
import torch 
import wandb

def main():
    print('Initializing model training...')
    device = torch.device('cpu')
    if torch.cuda.is_available(): 
        print('Running on gpu')
        device = torch.device('cuda')
    else:
        print('Running on cpu')


if __name__ == '__main__':
    main()
