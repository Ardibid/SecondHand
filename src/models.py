'''
    File name: models.py
    Author: Ardavan Bidgoli
    Date created: 10/13/2021
    Date last modified: 10/13/2021
    Python Version: 3.8.5
    License: MIT
'''

##########################################################################################
# Import
##########################################################################################

# ML
import torch
import torch.nn as nn

from torch import functional as F
from torch import optim as optim

# misc
import numpy as np
from os.path import join
# modules
from .utils import batch_recon

##########################################################################################
# Classes
##########################################################################################


class Encoder(nn.Module):
    """
    The Encoder model 
    """

    def __init__(self, opt, device):
        """
        The only input is the options
        """

        super(Encoder, self).__init__()
        self.device = device
        self.options = opt
        self.latent_size = self.options.LATENT_DIM

        # The encoder CNN layers follow the CMBL order:
        # Conv -> MaxPool -> Batch Normalization -> LeakyRelu
        # The first maxpooling layer is commented out to keep the shape correct
        # at the end

        self.cell_filter = [16, 32, 64, 128, 8]

        self.cnn_cell_1 = nn.Sequential(nn.Conv2d(1, self.cell_filter[0], 3, stride=1, padding=1),
                                        # nn.MaxPool2d(2),
                                        nn.BatchNorm2d(self.cell_filter[0]),
                                        nn.LeakyReLU(0.2))

        self.cnn_cell_2 = nn.Sequential(nn.Conv2d(self.cell_filter[0],  self.cell_filter[1], 3, stride=1, padding=1),
                                        nn.MaxPool2d(2),
                                        nn.BatchNorm2d(self.cell_filter[1]),
                                        nn.LeakyReLU(0.2))

        self.cnn_cell_3 = nn.Sequential(nn.Conv2d(self.cell_filter[1], self.cell_filter[2], 3, stride=1, padding=1),
                                        nn.MaxPool2d(2),
                                        nn.BatchNorm2d(self.cell_filter[2]),
                                        nn.LeakyReLU(0.2))

        self.cnn_cell_4 = nn.Sequential(nn.Conv2d(self.cell_filter[2], self.cell_filter[3], 3, stride=1, padding=1),
                                        nn.MaxPool2d(2),
                                        nn.BatchNorm2d(self.cell_filter[3]),
                                        nn.LeakyReLU(0.2))

        self.cnn_cell_5 = nn.Sequential(nn.Conv2d(self.cell_filter[3], self.cell_filter[4], 3, stride=1, padding=1),
                                        nn.MaxPool2d(2),
                                        nn.BatchNorm2d(self.cell_filter[4]),
                                        nn.LeakyReLU(0.2))

        self.flatting = nn.Flatten()
        self.fc = nn.Linear(128, self.latent_size*2)

        # this layer converts the size of labels to the size of images
        self.embedding = nn.Linear(52, self.options.X_DIM)

    def reparametrization(self, mean, log_var):
        """
        Samples from a normal distribution with a given set of
        means and log_vars
        """
        # epsilon is a vector of size (1, latent_dim)
        # it is samples from a Standard Normal distribution
        # mean = 0. and std = 1.
        epsilon = torch.normal(
            mean=0, std=1, size=log_var.shape).to(self.device)

        # we need to convert log(var) into var:
        var = torch.exp(log_var*0.5)

        # now, we change the standard normal distributions to
        # a set of non standard normal distributions
        z = mean + epsilon*var
        return z

    def forward(self, x):
        """
        Forward pass of the encoder
        x: input data of shape [batch_size, 64,64]
        """
        x = x.view(x.shape[0], 1, self.options.X_DIM, self.options.X_DIM)

        # the CNN cells
        x = self.cnn_cell_1(x)
        x = self.cnn_cell_2(x)
        x = self.cnn_cell_3(x)
        x = self.cnn_cell_4(x)
        x = self.cnn_cell_5(x)
        x = self.flatting(x)
        encoded = self.fc(x)

        # breaking the encoded layer into two tensors:
        # mu and logvar
        mean = encoded[:, : encoded.shape[-1]//2]
        log_var = encoded[:, encoded.shape[-1]//2:]

        # passing means and log_vars to the reparametrization function
        # to get corresponding samples from the normal distributions
        z = self.reparametrization(mean, log_var)

        return z, mean, log_var


class Decoder(nn.Module):

    def __init__(self, opt, device):
        """
        The decoder class is the same as one from an AutoEncoder
        """
        super(Decoder, self).__init__()

        self.options = opt
        self.device = device
        self.latent_size = self.options.LATENT_DIM

        self.fuser = nn.Sequential(nn.Linear(self.latent_size, 128))

        self.cell_filter = [8, 128, 64, 32, 16]

        self.convT_cell_1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.cell_filter[0], self.cell_filter[1], 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.cell_filter[1]),
            nn.LeakyReLU(0.2))

        self.convT_cell_2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.cell_filter[1], self.cell_filter[2], 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.cell_filter[2]),
            nn.LeakyReLU(0.2))

        self.convT_cell_3 = nn.Sequential(
            nn.ConvTranspose2d(
                self.cell_filter[2], self.cell_filter[3], 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(self.cell_filter[3]),
            nn.LeakyReLU(0.2))

        self.convT_cell_4 = nn.Sequential(
            nn.ConvTranspose2d(
                self.cell_filter[3], self.cell_filter[4], 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.cell_filter[4]),
            nn.LeakyReLU(0.2))

        self.convT_cell_5 = nn.Sequential(
            # , padding= (0,1)),
            nn.ConvTranspose2d(
                self.cell_filter[4], self.cell_filter[4], 3, stride=1),
            nn.BatchNorm2d(self.cell_filter[4]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.cell_filter[4], 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        ).to(self.device)

        self.matcher = nn.Sequential(
            nn.Linear(self.options.LATENT_DIM, 128),
            nn.LeakyReLU(0.2),
        )

    def forward(self, z):
        """
        z: the latent vector [-1, 64]
        """

        z = self.matcher(z)
        z = z.view(z.shape[0], 8, 4, 4)
        xy = self.convT_cell_1(z)
        xy = self.convT_cell_2(xy)
        xy = self.convT_cell_3(xy)
        xy = self.convT_cell_4(xy)
        x_rec = self.convT_cell_5(xy)

        return x_rec


class VAE (nn.Module):
    """
    A class to build a Conditional Variational AutoEncoder
    """

    def __init__(self, opt, device):
        """
        Creates the VAE model
        Only needs the options
        """
        super(VAE, self).__init__()
        self.options = opt
        self.device = device
        self.encoder_model = Encoder(self.options, self.device).to(device)
        self.decoder_model = Decoder(self.options, self.device).to(device)
        self.fusion = nn.Linear(opt.LATENT_DIM + opt.N_CLASSES, opt.LATENT_DIM)

    def forward(self, x, y):
        """
        It first passes the data and labels to the encoder to get z, mean, log_var
        then passes the z and a lable to the decoder to generate a conditioned x
        """
        z, mean, log_var = self.encoder_model(x)

        ########################################################################
        # Changes to convert the model to CVAE
        # preparing y as the condition signal
        y = y.squeeze()

        # combining z and y
        z_condition = torch.cat((z, y), 1)
        z_c_fused = self.fusion(z_condition)

        # send the fused signal to the decoder!
        x_rec = self.decoder_model(z_c_fused)
        # End of changes
        ########################################################################

        return x_rec, mean, log_var


def loss_function(x, x_rec, mean, log_var):
    """
    Calcualtes both loss functions for distribution and reconstruction
      x: original data
      x_rec: the reconstructed data
      mean: mean vector
      log_var: log_var vector
    """
    # reconstruction loss using Binary Cross Entropy
    rec_loss = nn.functional.binary_cross_entropy(
        x_rec.squeeze(), x.squeeze().detach(), reduction='sum')

    # the distribution loss using KLD
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # you can also add different weight to these two
    total_loss = rec_loss + KLD
    return total_loss, rec_loss, KLD


def eval_model(model, options, data_iterator, device, plot, epoch=0):
    """
    This function puts the model in eval mode (to change the behavior of batchNorm)
    and then pass the test data to evaluate the model
      model: model to evaluate
      data_iterator: usually the test_dataiterator
      device: 'cpu' or 'cuda0', to determine which computing device to use
      plot: defines if we need to plot the results
    """

    model.eval()
    # turning off the gradient
    with torch.no_grad():
        eval_losses = []
        eval_rec_losses = []
        eval_kld_losses = []

        # iterate over the data iterator
        for data in (data_iterator):
            x = data[0]
            x = x.to(device)
            y = data[1]
            y = y.to(device)

            # pass the data in the model
            x_rec, mean, log_var = model(x, y)

            # calcualte the loss
            loss, rec_loss, kld_loss = loss_function(
                x.squeeze(), x_rec.squeeze(), mean, log_var)

            eval_losses.append(loss.item())
            eval_rec_losses.append(rec_loss.item())
            eval_kld_losses.append(kld_loss.item())

    return eval_losses, eval_rec_losses, eval_kld_losses


##########################################################################################
# Support functions
##########################################################################################

def train_model(model, optimizer, train_iterator, test_iterator, device, options, plot_folder= None, file_names= None):
    """
    This function puts the model in training mode (to change the behavior of batchNorm)
    and then pass the train data to train the model
      model: model to evaluate
      train_iterator: data loader for training
      test_iterator: data iterator for evaluation and test
      device: 'cpu' or 'cuda0', to determine which computing device to use
      options: the model options
    """
    train_plot_numpy_file, training_history_numpy_file, validation_history_numpy_file = file_names

    # some placholders
    train_loss_history = []
    eval_loss_history = []
    eval_rec_losses_history = []
    eval_kld_losses_history = []

    fixed_sample = None
    
    training_history_numpy_file_path = join(plot_folder, training_history_numpy_file)
    np.save(training_history_numpy_file_path, np.zeros(shape=(1, 1)))
    
    validation_history_numpy_file_path = join(plot_folder, validation_history_numpy_file)
    np.save(validation_history_numpy_file_path, np.zeros(shape=(1, 1)))

    train_plot_numpy_file_path = join(plot_folder, train_plot_numpy_file)
    np.save(train_plot_numpy_file_path, np.ones(shape=(6*64, 8*64)))

    for epoch in range(options.N_EPOCHS):
        epoch_loss = []
        model.train()
        # iterate over the data iterator
        for data in (train_iterator):
            x = data[0]
            x = x.to(device)

            y = data[1]
            y = y.to(device)
            if fixed_sample is None:
                fixed_sample = [x, y]

            optimizer.zero_grad()

            # pass the data in the model
            x_rec, mean, log_var = model(x, y)

            # calcualte the loss
            train_loss, train_rec_losses, train_kld_losses = loss_function(
                x.squeeze(), x_rec.squeeze(), mean, log_var)

            # taking care of the next steps!
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())

        # plot the results once every a few epochs (to increase the speed)
        if (epoch % 5 == 0):
            plot = True
        else:
            plot = False
        # evaluating the model and measuring the eval loss
        eval_loss, eval_rec_losses, eval_kld_losses = eval_model(
            model, options, test_iterator, device, plot, epoch)
        train_loss_history.append(np.mean(epoch_loss))
        eval_loss_history.append(np.mean(eval_loss))

        print(epoch, train_loss.item(),
              train_rec_losses.item(), train_kld_losses.item())

        if plot:
            np.save(training_history_numpy_file_path, #"./plots/training_loss_history",
                    np.array(train_loss_history))
                    
            np.save(validation_history_numpy_file_path, #"./plots/validation_loss_history",
                    np.array(eval_loss_history))

        eval_rec_losses_history.append(np.mean(eval_rec_losses))
        eval_kld_losses_history.append(np.mean(eval_kld_losses))
        batch_recon(model, options, train_iterator, True, fixed_sample)

    return model, train_loss_history, eval_loss_history
