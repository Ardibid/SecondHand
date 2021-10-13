'''
    File name: sampling.py
    Author: Ardavan Bidgoli
    Date created: 10/13/2021
    Date last modified: 10/13/2021
    Python Version: 3.8.5
    License: MIT
'''

##########################################################################################
# Import
##########################################################################################
import numpy as np
import torch
import time
import string
import textwrap
from os.path import join


##########################################################################################
# Functions
##########################################################################################

def single_letter_test(char_index, mean, std,  model, data, char_data, device, items_to_show):
    """
    A function to fine-tune every character in the font. 
    inputs:
        char_index: the index of the character that you are curently working on
        mean: mean value for the sampling
        std" the standard deviation value for the sampling
        items_to_show : the number of generated samples for each character
        model: the VAE model
        data: the dataset to start sampling from 
        char_data (dict): data of all letters created
        device (string): device that should run the model
        items_to_show (int): number of items to generated (to the power of two)
    outputs:
        images (nparray): grid of rendered chars, shape (64*items_to_show, 64*items_to_show)
        char_data (dict): data of all letters created
        generated_ones (list): list of characters created
        remaining (list): lsit of characters remaining
    """
    # taking care of data and indices
    char_keys = string.ascii_letters

    char_index = int(char_index)
    char_requested = char_keys[char_index]

    indices = (
        np.where(np.where(data.Y.detach().cpu().numpy() == 1)[1] == char_index))
    # if the samples are more than what we need, it will randomly pick a subset of them
    rnd = np.random.randint(0, indices[0].shape[0], size=(items_to_show**2))
    indices = indices[0][rnd]

    sample_letters = data.X[indices]
    sample_labels = data.Y[indices]

    # send samples to the encoder model to get the latent vector
    z_vec, mean_vec, std_vec = model.encoder_model(sample_letters)
    # calculates the mean z vector as the start point for the generation
    z_vec = torch.mean(z_vec, axis=0).repeat(items_to_show**2, 1)

    # creating and adding a random value to the mean z vector to create variations
    random_factor = torch.normal(mean, std, size=z_vec.shape).to(device)
    tmp_zvec = z_vec + random_factor

    # sending the z and label to the decoder
    sample_label = torch.unsqueeze(
        sample_labels[0], 0).repeat(items_to_show**2, 1)
    z_condition = torch.cat((tmp_zvec, sample_label), 1)

    z_c_fused = model.fusion(z_condition)
    new_sample = model.decoder_model(
        z_c_fused).squeeze().cpu().detach().numpy()

    # converting the outputs to a big image
    images = (new_sample.reshape(items_to_show, items_to_show, 64, 64, 1)
              .swapaxes(1, 2)
              .reshape(64*items_to_show, 64*items_to_show, 1)).squeeze()

    # storing data in the char_data
    char_data[char_requested] = new_sample

    generated_ones = [key for key, val in char_data.items()]
    remaining = [c for c in char_keys if c not in generated_ones]

    return images, char_data, generated_ones, remaining


def finalize_font(char_data, render_folder, render_file_pattern):
    """
    Plots a simple grid for of one sample per letters that you have generated 
    in the previous cell
    inputs:
        char_data (dictionary): a dictionary of all samples that you have generated
    outputs:
        images (nparray): rendered chars shape (64*2, 64*26) which is (128, 1404)
    """

    char_keys = string.ascii_letters

    # finding the lowest number of samples for all characters
    dim = min(([char_data[key].shape[0] for key, val in char_data.items()]))
    # check for missing characters
    if dim == 0:
        print("You are missing some of the letters")

    # placeholder variables
    full_font_data = np.empty(shape=(52, dim, 64, 64))
    images = np.ones(shape=(64, 52*64))
    images = np.ones(shape=(64*2, 26*64))

    # sampling from the char_data and render the whole alphabet
    for counter, c in enumerate(char_keys):
        full_font_data[counter] = char_data[c][:dim]
        tmp = full_font_data[counter][np.random.randint(0, dim)][:, :]
        if counter < 26:
            images[:64, counter*64: (counter+1)*64] = tmp
        else:
            images[64:, (counter-26)*64: (counter-26+1)*64] = tmp

    # display and save
    timestr = time.strftime("%Y%m%d-%H%M%S")

    render_file_path = join(render_folder,(render_file_pattern+timestr+".npy"))
    np.save(render_file_path, #"./renders/full_font_data_{}".format(timestr), 
            full_font_data)

    return images


def render_text(text_input, char_data, squeeze_factor=20, line_padding=10):
    """
    This function uses the letters from the char data and renders the text
    inputs:
        text_input: type (or paste) your text in the text box and it will automatically renders it by your saved font.
        squeeze_factor: changes the gap between the letters
        line_padding: changes the space between the lines if you enter more than 50 characters.
    return:
        images (nparray): rendered text as a numpy array 
    """

    dim = min(([char_data[key].shape[0] for key, val in char_data.items()]))

    # taking care of long text in multiple lines
    line_len = 50
    lines = textwrap.wrap(text_input, line_len)
    n_lines = len(lines)

    images = np.ones(shape=((63+line_padding)*n_lines,
                           (64-squeeze_factor)*line_len))

    # rendering one char at a time as well as skipping special characters
    # and empty spaces
    for j, line in enumerate(lines):
        letter_in_line = 0
        for i, c in enumerate(line):
            valid_char = False
            tmp = None

            if (c == " "):
                tmp = np.ones(shape=(63, (64-squeeze_factor)))
                valid_char = True
            else:
                try:
                    tmp = char_data[c][0][:-1,
                                          squeeze_factor//2:-squeeze_factor//2]
                    valid_char = True
                except:
                    pass

            if valid_char:
                images[(63+line_padding)*j:(63+line_padding)*j+63, (64-squeeze_factor)
                      * letter_in_line:(64-squeeze_factor)*(letter_in_line+1)] = tmp
                letter_in_line += 1

    return images
