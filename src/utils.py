##########################################################################################
###### Import 
##########################################################################################
# ML
import torch
from torch.utils.data import DataLoader, Dataset

# misc
import cv2
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# update the x_path and y_path with the correct path in your Drive
class Handwiriting_dataset(Dataset):
    def __init__(self, x_path= None, y_path = None, split = 0.2 , resize_input = False):
        """
        A custom-made dataset class to handle 
        images and labels
        x_path= path to x numpy file
        y_path= path to labels files
        split = test set to all data ratio
        resize_input = if True, the x dataset will be resized to a given dimension
        """
        global main_path
        self.resize_input = resize_input

        # default paths
        if not x_path:
            x_path = main_path+"/database/main_data/alphabet_handwriting_64.npy"
        if not y_path:
            y_path = main_path+"/database/main_data/labels.npy"

        X, Y = self.read_data(verbuse= True, x_path = x_path, y_path= y_path)

        self.X = torch.from_numpy(X).float().to(device)
        self.Y = torch.from_numpy(Y).float().to(device)

    def read_data(self, x_path= None, y_path= None, verbuse= False):
        """
        Reads the X and Y data from a given path, 
        if no path is provided, it will use the default paths
        x_path= path to x numpy file
        y_path= path to labels files
        verbuse= if True, it plots further information
        """
        
        print (x_path) 
        X = np.load(x_path)
        Y = np.load(y_path)
        
        if verbuse:
            print ("X shape: {}\nY shape: {}".format(X.shape, Y.shape))
        

        if self.resize_input:
            # changes the input data into this size
            # this is designed to reduce the original 90x90
            # size to enhance the learning process

            self.new_size = 64
            print ("Resizing data to {}x{}".format(self.new_size,self.new_size))
            try:
                x_path = "./alphabet_handwriting_{}.npy".format(self.new_size)
                X = np.load(x_path)
                print ("X shape: {}\nY shape: {}".format(X.shape, Y.shape))
            except:
                new_data = np.ones((1, self.new_size, self.new_size))

                for img in X:
                    img_ = cv2.resize(img, (self.new_size,self.new_size))
                    new_data = np.append(new_data, img_.reshape(1, self.new_size, self.new_size), axis = 0)
                    
                new_data = new_data [1:]
                print (X.shape)
                print (new_data.shape)
                np.save("./alphabet_handwriting_{}.npy".format(self.new_size), new_data)
                X = new_data
            self.input_dim = X.shape[-1]

        return X, Y

    def __len__(self):
        """
        Default behavior when len is called
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Default behavior when an item of the dataset is being called
        """
        return [self.X[idx], self.Y[idx]] #, self.Y_[idx], self.XY[idx]] 


class Option_list(object):
    def __init__(self, n_class= 52):
        """
        A class to hold the options and settings for the VAE
        """
        self.BATCH_SIZE = 64          # number of data points in each batch
        self.N_EPOCHS = 50           # times to run the model on complete data
        self.LATENT_DIM = 128         # latent vector dimension
        self.N_CLASSES = n_class      # number of classes in the data
        self.lr = 1e-3                # learning rate
        self.LEARN_TEST_RATIO = 0.1   # test set to all data ratio
        self.X_DIM = 64               # the width/height of each sample


def dataloader_creator(raw_data, opt):
    """
    Convert the raw data into test/train dataloaders and iterators
    """
    training_size = int(raw_data.X.shape[0]*(1. - opt.LEARN_TEST_RATIO))
    test_size = raw_data.X.shape[0] - training_size
    validation_size = int(training_size*0.75)
    training_size -= validation_size
    
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(raw_data,[training_size,
                                                                          validation_size, 
                                                                          test_size])

    train_iterator = DataLoader(train_dataset, 
                                batch_size=opt.BATCH_SIZE,
                                shuffle=True, 
                                drop_last= True)
    
    validation_iterator = DataLoader(validation_dataset, 
                                batch_size=opt.BATCH_SIZE,
                                shuffle=True, 
                                drop_last= True)

    test_iterator = DataLoader(test_dataset, 
                                batch_size=opt.BATCH_SIZE, 
                                shuffle=True, 
                                drop_last= True)
    
    return train_dataset, validation_dataset, test_dataset, train_iterator, validation_iterator, test_iterator


def batch_recon(model, opt, train_iterator, save_on_disk = False, fixed_sample=None):
    """
    Adds the same value to all elements of z
    model: model to use
    z: latent vector for one instance
    char_index: constant value to add to z 
            char_index = 0 means a, 51 means Z
    """
    
    with torch.no_grad():
        for data in (train_iterator):
            
            if fixed_sample is None:
                x = data[0]
                x = x.to(device)
                y = data[1]
                y = y.to(device)
            else:
                x, y = fixed_sample

            z, mean, log_var = model.encoder_model(x) 
            break
    dim = opt.X_DIM
    #items_to_display = 32
    
    rows = 2
    cols= 8
    border_line = 2

    items_to_display = rows* cols

    z = z[:items_to_display,:]
    y = y[:items_to_display,:]

    z_condition = torch.cat((z, y), 1) 

    z_c_fused = model.fusion(z_condition)
    new_sample = model.decoder_model(z_c_fused).squeeze().cpu().detach().numpy()
    x = x.squeeze().cpu().detach().numpy()

    x = np.clip(x, 0, 1)
    new_sample = np.clip(new_sample, 0, 1)

    image = np.ones((dim*rows*3, dim*cols))
    
    for i in range (rows):
        for j in range (cols):
            index = i*cols + j
            image[ 3*i*dim: (3*i+1)*dim, j*dim: (j+1)*dim] = x[index][:]
            image[ (3*i+1)*dim: (3*i+2)*dim, j*dim: (j+1)*dim] = new_sample[index][:]
            image[ (3*i+2)*dim: (3*i+3)*dim, j*dim: (j+1)*dim] = x[index][:] - new_sample[index][:]
    
    for i in range(1, rows):
        image [(dim*i*3)-border_line:dim*i*3+border_line, : ] = np.zeros((2*border_line, dim*cols))

    
    if save_on_disk:
        np.save("./plots/train_plot", image)

