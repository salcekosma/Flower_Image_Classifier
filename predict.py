#!/usr/bin/python

# Imports here
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import nn, optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import argparse

#Loading checkpoint
def load_checkpoint(saved_checkpoint):
    checkpoint = torch.load(saved_checkpoint)
    state_dict = checkpoint['state_dict']
    mm = checkpoint['model']
    mm.load_state_dict(state_dict)
    return mm
model = load_checkpoint('model_checkpoint.pth')

#image processing
def process_image(image):
    image_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    img_pil = Image.open(image)
    img_tensor = image_transform(img_pil)
    numpy_image = np.array(img_tensor)

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    numpy_image = (np.transpose(numpy_image,(1,2,0))-mean)/std
    numpy_image = np.transpose(numpy_image, (2,0,1))

    return numpy_image

def predict(args, model):
    model.eval()
    image = process_image(args.image_path)
    image_tensor = torch.FloatTensor([image])
    idx_to_class = torch.load(args_checkpoint)['class_to_idx']
    idx_to_class = {i: j for j, i in idx_to_class.items()}
    if gpu_available and args.gpu:
        image_tensor.cuda()
        model.cuda()
    else:
        image_tensor.cpu()
        model.cpu()
    output = model.foward(image_tensor)
    ps = torch.exp(output).data[0]
    class_index = ps.topk(args.topk)
    classes = [idx_to_class[i] for i in class_index[1].cpu().numpy()]
    probabilities = class_index[0].cpu().numpy()
    return probabilities, classes

def main():
    parser = argparse.ArgumentParser(description='Neural Network Settings')

    parser.add_argument('--topk',
                        type = int,
                        default = 5,
                        help = 'topk returns top results')

    parser.add_argument('--checkpoint',
                        type = str,
                        default = 'model_checkpoint.pth',
                        help = 'Path to checkpoint')

    parser.add_argument('--gpu',
                        type = bool,
                        action = 'store_true',
                        default = False,
                        help = 'Use GPU + CUDA for calculations')

    parser.add_argument('--image_path',
                        type = str,
                        help = 'Image path')

    parser.add_argument('--jsonpath',
                        type = str,
                        default = 'cat_to_name.json',
                        help = 'Path to json')

    args = parser.parse_args()

    with open(args.jsonpath, 'r') as file:
        cat_to_name = json.load(file)

    model = load_checkpoint(args.checkpoint)

    probabilities, classes = predict(args, model)

    print('classes: ', classes) [print(cat_to_name[x]) for x in classes] print('probabilities: ', probabilities)

if __name__ == "__main__":
    main()
