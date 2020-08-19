from model import FeatureExtractor
from sklearn.preprocessing import normalize
import torch.nn.functional as functional
import torchaudio
import argparse
import torch
import sys

def get_parser():
    parser = argparse.ArgumentParser(description='Test the generative network')
    parser.add_argument('--model', type=str, help='Path to the pretrained model')
    parser.add_argument('--input', type=str, help='Path to the input to test')
    parser.add_argument('--output', type=str, help='Path to the output sound')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    model_path = args.model
    input_path = args.input
    sound_path = args.output
    model = FeatureExtractor()
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda')
    cpu_device = torch.device('cpu')
    model.to(device)
    #data = normalize(torchaudio.load(input_path)[0][0].reshape(1, -1))
    data = torch.from_numpy(normalize(torch.randn(1, 132480))).float().to(device)
    data = data.reshape(1, 1, -1)
    model.eval()
    sound = model(data)
    print(functional.mse_loss(sound, data).item())
    sound = sound.to(cpu_device)
    torchaudio.save(sound_path, sound.reshape(-1), 44100)

if __name__ == '__main__':
    main()
