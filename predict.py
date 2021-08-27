"""
Predict flower name from an image along with the probability of that name.
Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""
import argparse
import sys
import torch
import numpy as np
import json

from utilities import process_image, load_data, image_for_predict
from model_func import load_checkpoint


def get_input_args():
    """
        Retrieves and parses command line arguments provided by the user when
        they run the program from a terminal window. This function uses Python's 
        argparse module to create and define these command line arguments. If 
        the user fails to provide some or all of the arguments, then the default 
        values are used for the missing arguments. 
        Command Line Arguments:
          1. input which rep to /path/to/image 
          2. checkpoint
          3. top probabilities as --topk with default value 3
          4. map cat to real names --category_names with default cat_to_name.json
          5. device type default= "cpu"
        This function returns these arguments as an ArgumentParser object.
        Parameters:
        Returns:
         parse_args() -data structure that stores the command line arguments object  
        """
    # print(sys.argv[1])
    while True:
        try:
            if sys.argv[1] and sys.argv[2]:
                if "--gpu" in sys.argv[1:]:
                    print("use gpu for inference")
                else:
                    print("use cpu for inference")
                break
        except IndexError:
            print("input an image path")
            print("input checkpoint")
            return
            # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 4 command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('data_dir')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()


in_arg = get_input_args()

# label mapping
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

# calling saved model
saved_model, saved_checkpoint = load_checkpoint(in_arg.checkpoint)


def predict(image_path, model, top_num):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    train_data = load_data(in_arg.data_dir)

    model_input = image_for_predict(image_path)

    # probs
    probs = torch.exp(model.forward(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    train_data.class_to_idx.items()}
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs] #list of numbers 3 6 9

    # # getting actual name of flower
    # flower_num = image_path.split('test/')[1]
    # flower_num = flower_num.split('/')[0]
    # flower_name = cat_to_name[flower_num]
    # print(f'The actual flower name is: {flower_name}')

    return top_probs, top_flowers


probs, classes = predict(in_arg.image_path, saved_model, in_arg.topk)
print(probs)
print(classes)
