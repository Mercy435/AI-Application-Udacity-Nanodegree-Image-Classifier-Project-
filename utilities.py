"""
Loading and preprocessing data
"""
import torch
import numpy as np

from torchvision import datasets, transforms, models
from PIL import Image


def load_data(data_dir):
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    data_transforms = {"train_transforms": train_transforms, " valid_transforms": valid_transforms,
                       "test_transforms": test_transforms}

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + "/valid", transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)

    image_datasets = {"train": train_data, "valid": valid_data, "test": test_data}

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    dataloaders = {"trainloader": trainloader, "validloader": validloader, "testloader": testloader}

    return trainloader, validloader, testloader, train_data


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model

    # load image
    image = Image.open(image_path)

    # transform image
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    image = img_transforms(image)

    # converting to color channels by converting to numpy array
    np_image = np.array(image)
    image = np_image

    # print(image.shape)
    return image


def image_for_predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = process_image(image_path)
    img = img.to(device)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image so shape of image is (1,3,224,224)==(batch_size, rgb, width, height)
    model_input = image_tensor.unsqueeze(0)

    return model_input
