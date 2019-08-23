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
import json
from collections import OrderedDict
import argparse

gpu_available = torch.cuda.is_available


def data_preparation(args):

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    data_sets= {}
    data_sets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    data_sets['validation'] = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    data_set['test'] = datasets.ImageFolder(test_dir ,transform = test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    data_loaders = {}
    data_loaders['train'] = torch.utils.data.DataLoader(data_sets['train'], batch_size=64, shuffle=True)
    data_loaders['validation']= torch.utils.data.DataLoader(data_sets['validation'], batch_size =64,shuffle = True)
    data_loader['test'] = torch.utils.data.DataLoader(data_sets['test'], batch_size = 64, shuffle = True)

    return data_loaders, data_sets

def training_function(args, steps, optimizer, print_every, criterion, model):
    if gpu_available and args.gpu:
        model.to('cuda')

    data_loaders, data_sets = data_preparation(args)
    train_data_loaders = data_loaders['train']
    validation_data_loaders = data_loaders['validation']

    for epoch in range(args.epochs):
        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_data_loaders:
            steps += 1
            if gpu_available and args.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() #updates weights automatically

            running_loss += loss.item()

            if steps % print_every == 0:
                running_valid_loss = 0
                valid_total = 0
                valid_correct = 0
                model.eval()

                for inputs, labels in validation_data_loaders:
                    if gpu_available and args.gpu:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = model.forward(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    running_valid_loss += criterion(outputs, labels).item()
                    valid_total += labels.size(0)
                    valid_correct += (predicted == labels).sum().item()

                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every),
                      "Train Accuracy: %d %%" % (100 * correct / total)
                     )
                print('-------------------------------------------')
                print("Valid Loss: {:.4f} ...".format(running_valid_loss/print_every),
                      "Valid Accuracy: %d %%" % (100 * valid_correct / valid_total)
                     )
                print('********************************************')
                print()

                running_loss = 0
    return model

def main():

    parser = argparse.ArgumentParser(description='Neural Network Settings')

    parser.add_argument('--arch',
                        type = str,
                        action = 'store',
                        default = 'vgg',
                        help = 'Choose architecture')

    parser.add_argument('--save_dir',
                        type = str,
                        action = 'store',
                        default = 'model_checkpoint.pth',
                        help = 'Define save directory for checkpoint')

    parser.add_argument('--learning_rate',
                        type = float,
                        action = 'store',
                        help = 'Define gradient descent learning rate (lr)')

    parser.add_argument('--epochs',
                        type = int,
                        action = 'store',
                        help = 'Number of epochs for training')

    parser.add_argument('--gpu',
                        type = bool,
                        action = 'store_true',
                        default = False,
                        help = 'Use GPU + CUDA for calculations')

    parser.add_argument('--hidden_units',
                        type = int,
                        action = 'store_true',
                        default = 100,
                        help = 'Hidden Units')

    parser.add_argument('--data_dir',
                        type = str,
                        default = 'flowers',
                        help = 'Data Directory')

    args = parser.parse_args()


    #Label mapping

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)


    #Build and train your network
    if args.arch == 'vgg':
        model == models.vgg19(pretrained=True)
        input_features = model.classifier[0].in_features
        output_features = model.classifier[0].out_features
    elif args.arch == 'densenet':
        model == models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
        output_features = model.classifier.out_features


    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, output_features)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('hidden_units', nn.Linear(output_features, args.hidden_units)),
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))


    #Feedfoward classifier

    model.classifier = classifier
    if gpu_available and args.gpu:
        model.cuda()

criterion = nn.NLLLoss
optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
model = training_function(args, steps = 0, optimizer, print_every, criterion, model)
data_loaders, data_sets = data_preparation(args)
model.class_to_idx = data_sets['train'].class_to_idx

checkpoint = {
    'epochs': epochs,
    'batch_size': 102,
    'classifier': classifier,
    'model': model,
    'optimizer': optimizer.state_dict(),
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'criterion': nn.NLLLoss()
    }

torch.save(checkpoint, 'model_checkpoint.pth')

if __name__ == "__main__":
    main()
