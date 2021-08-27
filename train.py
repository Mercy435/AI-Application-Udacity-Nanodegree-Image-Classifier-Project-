"""
Train a new network on a data set
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
"""

import argparse
import sys
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from utilities import load_data
from workspace_utils import active_session
from model_func import pretrain, build_model


def set_input_args():
    """
        Retrieves and parses command line arguments provided by the user when
        they run the program from a terminal window. This function uses Python's 
        argparse module to create and define these command line arguments. If 
        the user fails to provide some or all of the arguments, then the default 
        values are used for the missing arguments. 
        Command Line Arguments:
          1. data directroy as data_dir
          2. directory to save checkpoints as --savedir with default "save_directory"
          3. CNN Model Architecture as --arch with default value 'resnet152'
          4. hyper parameter --learning_rate with default value 0.001
          5. hyper parameter --hidden_units with default value 512 
          6. hyper parameter --epochs with default value 1
          4. device type default= "cpu"
        This function returns these arguments as an ArgumentParser object.
        Parameters:
        Returns:
         parse_args() -data structure that stores the command line arguments object  
        """
    # print(sys.argv[1])
    while True:
        try:
            if sys.argv[1]:
                if "--gpu" in sys.argv[1:]:
                    print("use gpu for training")
                else:
                    print("use cpu for training")
                break
        except IndexError:
            print("input a data directory")
            break
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='saved_checkpoints')
    parser.add_argument('--arch', default='resnet152')
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()


in_arg = set_input_args()
# print(in_arg)
# print(in_arg.data_dir)
data_dir = in_arg.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Load data
trainloader, validloader, testloader, train_data = load_data(data_dir)

hidden_layers = in_arg.hidden_units

# defining model
model = build_model(in_arg.arch, in_arg.hidden_units)

# Only train the classifier parameters, feature parameters are frozen

# defining loss
criterion = nn.NLLLoss()

if in_arg.arch == "resnet152":
    optimizer = optim.Adam(model.fc.parameters(), lr=in_arg.learning_rate)
elif in_arg.arch == "vgg16":
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)


def train():
    """
    trains model with user/default inputs, Prints out epochs, training loss, validation loss,
    and validation accuracy as the network trains

    """
    # training the model

    from workspace_utils import active_session

    with active_session():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = in_arg.epochs
        steps = 0
        running_loss = 0
        print_every = 5

        for epoch in range(epochs):
            for images, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy = 0

                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        # calculate the accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f" Valid loss: {valid_loss / len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy / len(validloader):.3f}")
                running_loss = 0
                model.train()


train()


# Save the checkpoint 
def save_checkpoint(model, file_path):
    if in_arg.arch == "resnet152":
        checkpoint = {'architecture': in_arg.arch,
                      'input_size': model.fc[0].in_features,
                      'hidden_layers': in_arg.hidden_units,
                      'output_size': 102,
                      'learning_rate': in_arg.learning_rate,
                      'epochs': in_arg.epochs,
                      'state_dict': model.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'classifier': model.fc,
                      'class_to_idx': train_data.class_to_idx}

    elif in_arg.arch == "vgg16":
        checkpoint = {'architecture': in_arg.arch,
                      'input_size': model.classifier[0].in_features,
                      'hidden_layers': in_arg.hidden_units,
                      'output_size': 102,
                      'learning_rate': in_arg.learning_rate,
                      'epochs': in_arg.epochs,
                      'state_dict': model.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'classifier': model.classifier,
                      'class_to_idx': train_data.class_to_idx}

    print("Saving model...")
    torch.save(checkpoint, file_path)


save_checkpoint(model, './' + in_arg.save_dir + "/" + in_arg.arch + "_checkpoint.pth")
