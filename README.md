## Udacity project-- AI APPLICATION (an image classifier)

This project is a trained image classifier; it accepts images of flowers and classifies them into their different 
breeds.

It makes use of CNN(Convolutional neural network).

The models used for these classifications are: Resnet152 and VGG16

The `train.py` script trains on a set of images from a data folder using a pretrained network, and saves the trained 
model as a checkpoint in a _save dir_

The `predict.py script accepts an _image path_ and the _saved model_, then returns the top probabilities as well as the 
names of the top flowers.

The `cat_to_name.json` file contains the labels and names as a dictionary

The `model_func.py` contains functions and classes relating to the model

The `utilities.py` contains functions for loading and preprocessing data

### Running the AI application on jupyter notebook:

1. import all the required modules for the project on the first cell

2. define the file paths of the images by assigning them to variables
   
3. load the data using `torchvision`, define the transforms as appropriate 
   
4. map category labels to category names from the json file `cat_to_name.json` containing the labels and names

5. build and train a classifier; load a pretrained model, freeze the model parameters and redefine the classifier to 
   suit your output, define the optimizer and loss function.
   Train the model defining the number of `epochs` as it prints the train loss, valid loss and valid accuracy after each
   epoch. Proper training should give an accuracy of at least 70%.
   
6. test the trained network on the test data; an accuracy of about 70% shows the model is apt.

7. save the trained network as a checkpoint
   
8. load the checkpoint to be used as inference
   
9. define a _process_image_ function that accepts `image path` (preferably test images) as input and preprocesses it to fit the format of 
   the images for which the saved network was trained on 
   
10. define a _predict_ function which accepts `image path`, `model` and `top_num` as inputs and returns `top_predictions`, 
    `top_labels` and 'top_flowers`.
    
11. finally, define a _plot_solution_ function which accepts `model` and `image path` as inputs and gives a visual 
    representation of predictions made by the trained network. 
    
### Running the AI application using the command line:

 The `argparse` module is the best way to get the command line input into the scripts 

1. using the command prompt or terminal, train a network by running the basic command `python train.py data_directory`. 
   This prints out training loss, validation loss, and validation accuracy as the network trains
   
2. `train.py` allows flexibility as users can give certain inputs such as:
   * _set directory_ to save checkpoints: python train.py data_dir --save_dir save_directory
   * _choose architecture_ : python train.py data_dir --arch "vgg13"
   * _set hyperparameters_: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
   * _use GPU for training_: python train.py data_dir --gpu
   
There are default values for these parameters if users do not specify them

3. Predictions are made running `python predict.py /path/to/image checkpoint` on the terminal.
   This returns the flower name and class probability
   
4. `predict.py` allows users to make some inputs, else use the defaults. 
   User inputs include:
   * _top K_ most likely classes: python predict.py input checkpoint --top_k 3
   * _Use a mapping of categories to real names_: python predict.py input checkpoint --category_names cat_to_name.json
   * _Use GPU for inference_: python predict.py input checkpoint --gpu
   
