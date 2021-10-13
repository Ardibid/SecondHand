'''
    File name: secondhand_dashboard.py
    Author: Ardavan Bidgoli
    Date created: 10/13/2021
    Date last modified: 10/13/2021
    Python Version: 3.8.5
    License: MIT
'''

##########################################################################################
###### Imports
##########################################################################################
# Dash and UI
import dash
from dash import dcc, html 
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ML
from torch import optim as optim

import numpy as np
import pandas as pd

from scipy.spatial import KDTree
from openTSNE import TSNE

# System
from os import listdir
from os.path import isfile, join

# Modules
from src.utils import *
from src.models import *
from src.sampling import *

##########################################################################################
###### Main setup
##########################################################################################
shared_dataset_labels = None
shared_dataset = None
show_visualization = False
merged_dataset = None
tsne_samples_size = 2500
tsne_data = None

# training global variables
number_of_epochs = 25
raw_data = None
train_dataset = None
test_dataset = None
train_iterator = None
test_iterator  = None
options = None
vae_model = None
optimizer = None
is_training = False    
char_data = {} 

num_of_classes= 52
sample_size_each_char = 36
sample_size = num_of_classes*sample_size_each_char

# path
data_folder = "./data/"
plot_folder = "./plots/"
model_folder = "./models/"
render_folder = "./renders/"

# file names
samples_file_name_pattern = "alphabet_handwriting_64_"

data_file_path = "alphabet_handwriting_64.npy"
data_label_file_path = "labels.npy"

tsne_distribution_numpy_file = "tsne_data.npy"

selected_data_file = "selected_data.npy"
selected_labels_file = "selected_labels.npy"

train_plot_numpy_file = "train_plot.npy"
training_history_numpy_file = "training_loss_history.npy"
validation_history_numpy_file = "validation_loss_history.npy"

saved_model_file = "trained_model"


##########################################################################################
###### App setup
##########################################################################################
external_stylesheets=[dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

##########################################################################################
###### UI Elements
##########################################################################################

controls = dbc.Card([
                    dbc.CardBody([
                                html.H4("Sample size", className="card-title"),
                                html.Br(),
                                dcc.Slider(id="sample_size",
                                            min = 100,
                                            max = 10000,
                                            value =3744,
                                            step = 36,
                                                    ),    
                                dbc.Label(id="sample_size_status", children= "Sample size"),
                                dbc.Label(id='selected_indices'),
                                html.Hr(),
                                ]),
                    dbc.CardBody([
                                html.H4("Control Panel", className="card-title"),
                                dbc.Button(id='merge_data', children= "Merge Data", color="dark", className="mr-1" ),
                                html.Br(),
                                dbc.Label(id='mrege_data_status', children="Data merge"),
                                html.Hr(),
                                dbc.Button(id='tsne_button', children= "t-SNE Proc.", color="dark", className="mr-1" ),
                                html.Br(),
                                dbc.Label(id='tsne_status'),
                                html.Br(),
                                html.Br(),
                                dbc.Spinner(html.Div(id="loading_output")),
                                ]),
                    ],
                    body=True,
                    style={"width": "w-20"},)

training = dbc.Card([
                    html.H4("Settings"),
                    html.Br(),
                    dcc.Slider(id="epoch_number",
                                    min = 1,
                                    max = 250,
                                    value = 150,
                                    step = 5,
                                ),
                    dbc.Label(id="epoch_number_status", children= "Number of Epochs: "),
                    html.Hr(),
                    dbc.Button(id='training_model', children= "Train Model", color="dark", className="mr-1" ),
                    html.Br(),
                    html.P(id="training_status", children="Trains on the selected data"),
                    html.P(id="validation_status", children="Current Selection size:"),
                    html.Hr(),
                    dbc.Button(id='save_model', children= "Save Model", color="dark", className="mr-1" ),
                    html.P(id="save_model_status", children=""),
                    html.Hr(),
                    dbc.Button(id='reset_model', children= "Reset Model", color="dark", className="mr-1" ),
                    html.P(id="reset_model_status", children=""),
                    dbc.Spinner(html.Div(id="training_output")),                                
                    ],
                    body=True,
                    style={"width":"w-20"},)

progress = dbc.Card([
                    dbc.CardBody([
                                html.H4("Training Progress"),
                                html.Br(),
                                dcc.Graph(id= "train_progress", style={'visibility':'hidden'}),
                                dcc.Graph(id="train_progress_graph", style={'visibility':'hidden'}),
                                dcc.Interval(id='interval-component', interval=2*1000,n_intervals=0), 
                                ])
                    ],
                    body=True,
                    style={"width": "w-70"},)

generation_graph = dbc.Card([
                            dbc.CardBody([
                                        html.H4("Generated Samples"),
                                        html.Br(),
                                        dcc.Graph(id= "generated_samples", style={'visibility':'hidden'}),
                                        ])
                            ],
                            body=True)


generation_variabels = dbc.Card([
                    dbc.CardBody([
                                    html.H4("Variables"),
                                    html.Br(),
                                    dbc.Label(id="char_index_label", children= "Character"),
                                    dcc.Slider(id="char_index",
                                                min = 0,
                                                max = 51,
                                                value =0,
                                                step = 1,
                                                updatemode='drag',
                                                ),
                                    dbc.Label(id="mean_label", children= "Mean value"),
                                    dcc.Slider(id="mean",
                                            min = -2.,
                                            max = 2.,
                                            value =0,
                                            step = 0.01,
                                            updatemode='drag',
                                        ),
                                    dbc.Label(id="std_label", children= "Std value"),
                                    dcc.Slider(id="std",
                                            min = 0.01,
                                            max = 1,
                                            value =0.01,
                                            step = 0.01,
                                            updatemode='drag',
                                        ),  
                                    html.P(id="generated_chars"),
                                    html.P(id="remaining_chars"), 
                                    dbc.Button(id='save_typeface', children= "Save Typeface", color="dark", className="mr-1" ),
                                    html.P(id="save_typeface_status", children=""),
                                    html.Hr(),
                                    dbc.Button(id='load_model', children= "Load Trained Model", color="dark", className="mr-1" ),
                                    html.Br(),
                                    html.P(id="spacer", children=""),
                                    dbc.Button(id='load_favorite_model', children= "Load Favorite Model", color="dark", className="mr-1" ),
                                    html.P(id="load_model_status", children=""),
                                    ])
                    ],
                    body=True,
                   )

text_render = dbc.Card([
                    dbc.CardBody([
                                    html.H4("Test Render"),
                                    html.Br(),
                                    dcc.Graph(id= "latest_typeface", style={'visibility':'hidden'}),
                                    dbc.Input(id="text_input", placeholder="Type something", type="text"),
                                    dbc.Label(id="squeeze_factor_status", children= "Squeeze Factor"),
                                    dcc.Slider(id="squeeze_factor",
                                            min = 15,
                                            max = 35,
                                            value =20,
                                            step = 1,
                                            updatemode='drag',
                                        ), 
                                    dbc.Label(id="line_padding_status", children= "Line Padding"),
                                    dcc.Slider(id="line_padding",
                                            min = 15,
                                            max = 35,
                                            value =20,
                                            step = 1,
                                            updatemode='drag',
                                        ), 
                                    dcc.Graph(id="test_render", style={'visibility':'hidden'}),
                                ])
                    ],
                    body=True,
                   )

operation_tabs = dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab([
                                                html.Br(),
                                                html.H3("Select and Mix your data"),
                                                html.P("USe the two plots above to select and mix any samples that you want"),   
                                                dbc.Col(controls, width=4)
                                                                                   
                                            ], 
                                            label="Curate", 
                                            tab_id="curation_tab"),

                                    dbc.Tab([
                                                html.Br(),
                                                html.H3("Train Model"),
                                                html.P("Train your machine learning model with the selected data"),
                                                dbc.Row(
                                                        [ 
                                                            dbc.Col(training, width=4),
                                                            dbc.Col(progress, width=8),
                                                        ]),
                                            ],
                                            label="Train", 
                                            tab_id="training_tab",),  
                                    
                                    dbc.Tab([
                                            html.Br(),
                                            html.H3("Generate Typeface"),
                                            html.P("Draw samples from your trained model"),
                                            dbc.Row(
                                                    [ 
                                                        dbc.Col(generation_variabels, width=4),
                                                        dbc.Col(generation_graph, width=8),
                                                    ]),
                                            html.Br(),
                                            dbc.Row(
                                                [ 
                                                dbc.Col(text_render, width=12),
                                                ], 
                                                )
                                            ],
                                            label="Generate", 
                                            tab_id="generating_tab",),       
                                ],
                                id="card_tabs",
                                card=True,
                                active_tab="curation_tab",
                            )

                        ),
                        dbc.CardBody(html.P(id="card-content", className="card-text")),
                    ]
                )

data_graphs = dbc.Card([
                        dbc.CardBody([
                                    dbc.Row(
                                            [ 
                                            dbc.Col(dcc.Graph(id= "data_plot"), className="five columns", style={'visibility':'hidden'}),
                                            dbc.Col(dcc.Graph(id= "data_plot_label"), className="five columns", style={'visibility':'hidden'}),
                                            ],
                                            align="center"
                                        ),
                                    dbc.Row(
                                            [   
                                            dbc.Col(dcc.Graph(id= "hover_data_fig"), className="three columns", style={'visibility':'hidden'}, width=6),
                                            dbc.Col(dcc.Graph(id= "selected_data_fig"), className="three columns", style={'visibility':'hidden'}, width=6),
                                            ],
                                            align="center"
                                            ),
                                    ])
                        ])
header = html.H1("SecondHand Dashboard")

app.layout = dbc.Container([
                            header,
                            html.Hr(),
                            data_graphs,
                            html.Hr(),
                            operation_tabs,
                        ])

##########################################################################################
###### Generate functions
###### All functions related to sampling from the ML model and generaing new typefaces
##########################################################################################
@app.callback(
    Output('test_render','figure'),
    Output('test_render', 'style'),
    Output('squeeze_factor_status', 'children'),
    Output('line_padding_status', 'children'),
    Input('text_input', 'value'),
    Input('squeeze_factor', 'value'),
    Input('line_padding', 'value'),
    prevent_initial_call=True)
def output_text(text_input, squeeze_factor, line_padding):
    """
    Renders an input text with the generated typeface.
    Inputs: 
        text_input (string): text to be rendered
        squeeze_factor (int): space between each letter
        line_padding (int): space between lines
    outputs:
        test_render figure (px.imshow): the rendered image
        test_render style (dictionary): converts the figure style to visible
        squeeze_factor_status (string): updates the text 
        line_padding_status (string): updates the text
    """
    global char_data
    
    if (text_input is None):
        return (dash.no_update, )*4
    
    else:
        # generating the rendered image
        images = render_text (text_input, char_data, squeeze_factor, line_padding)
        sample_fig = px.imshow(images, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)

        # formatting messages
        squeeze_factor_msg = ("Squeeze Factor: {}".format(squeeze_factor))
        line_padding_msg = ("Line padding: {}".format(line_padding))

        # saving image
        timestr = time.strftime("%Y%m%d-%H%M%S")
        sample_fig.write_image(render_folder+"rendered_text_{}.jpeg".format(timestr))

        return  (sample_fig, 
                {'visibility':'visible'}, 
                squeeze_factor_msg, 
                line_padding_msg)


@app.callback(
    Output("char_index_label", 'children'),
    Output("mean_label", 'children'),
    Output("std_label", 'children'),
    Output("generated_chars", 'children'),
    Output("remaining_chars", 'children'),
    Output("generated_samples",'figure'),
    Output('generated_samples', 'style'),
    Input('char_index', 'value'),
    Input('mean', 'value'),
    Input('std', 'value'),
    prevent_initial_call=True
    )
def generate_samples(char_index, mean, std):
    """
    Using user inputs to generate each character.
    Inputs: 
        char_index (int): index of each char, from 0 to 51
        mean (float): mean value for generation
        std (float): std value for generation
    outputs:
        char_index_label (string): updates char index text
        mean_label (string): updates mean text
        std_label (string): updates std text
        generated_chars (string): updates generated cahrs list text
        remaining_chars (string): updates remaining cahrs list text
        generated_samples (px imshow): image of generated sample at the moment
        generated_samples (dict): converts the figure style to visible

    """
    global char_data, vae_model, raw_data, device

    char = string.ascii_letters[int(char_index)]

    if raw_data is None:
        return ("Char: {}".format(char), 
                "Mean: {}".format(mean),
                "Std: {}".format(std), 
                dash.no_update,
                dash.no_update,
                dash.no_update, 
                dash.no_update, 
                )
    else:
        # generating the new sample grid
        images, char_data, generated_ones, remaining = single_letter_test(char_index, mean, std, vae_model, raw_data, char_data, device, 4)
        sample_fig = px.imshow(images, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)

        # generating messages
        generated_msg = ("{} chars generated: {}".format(len(generated_ones), generated_ones))
        remaining_msg = ("{} chars remaining: {}".format(len(remaining),remaining))

        return ("Char: {}".format(char), 
                "Mean: {}".format(mean),
                "Std: {}".format(std), 
                generated_msg,
                remaining_msg,
                sample_fig, 
                {'visibility':'visible'}, 
                )


@app.callback(
        Output("load_model_status", 'children'),
        Input("load_model", "n_clicks"),
        Input("load_favorite_model", "n_clicks"))
def load_saved_model(load_model, load_favorite_model):
    """
    Loads the default saved pytorch model. 
    inputs: 
        load_model n_clicks (int): checks if the button is pressed
    outputs:
        load_model_status children (string): updates the status text
    """
    global vae_model

    file_to_load = None
    msg = None
    ctx= None

    ctx = dash.callback_context


    # ignoring if the key has never pressed
    if not ctx.triggered:
        return (dash.no_update)
    
    # case for each button
    elif not( load_model is None and load_favorite_model is None):
        # checking what to load
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
       
        if button_id == "load_model":
            file_to_load = "./models/trained_model"
            msg = "Trained model is loaded"
        
        elif button_id == "load_favorite_model":
            file_to_load = "./models/favorite_model"
            msg = "Favorite model is loaded"
        
        else:
            return "Nof file loaded"

        # sampling requires having a dataloder, loads the latest selection
        load_training_data()
        vae_model = torch.load(file_to_load) 
        vae_model.eval()
        return msg
    
    # if there is no event
    else:
        return (dash.no_update)

@app.callback(
        Output("save_typeface_status", 'children'),
        Output("latest_typeface",'figure'),
        Output('latest_typeface', 'style'),
        Input("save_typeface", "n_clicks"))
def save_typeface(n_clicks):
    """
    Saving the generated typeface at the default path. Then shows a sample of all chars
    inputs:
        save_typeface n_clicks (int): checks if the button is pressed
    outpust:
        save_typeface_status children (string): updates the statues text
        latest_typeface figure (px.imshow): image of generated sample at the moment
        latest_typeface style (dictionary):  converts the figure style to visible
    """
    global char_data

    # ignoring if the key has never pressed
    if n_clicks is None:
        return (dash.no_update)
    else:
        # generating sample chars
        images = finalize_font(char_data)
        sample_fig = px.imshow(images, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)

        # saving the image
        timestr = time.strftime("%Y%m%d-%H%M%S")
        sample_fig.write_image("./renders/font_catalogue_{}.jpeg".format(timestr))
        
        return ("Typeface saved", 
                sample_fig, 
                {'visibility':'visible'})

##########################################################################################
###### Train functions
###### All functions related to trainig the ML model
##########################################################################################

@app.callback(Output('train_progress', 'figure'),
              Output('train_progress', 'style'),
              Output('train_progress_graph', 'figure'),
              Output('train_progress_graph', 'style'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    """
    Updates the training graphs automatically.
    """
    global is_training

    # check if the model is in training mode
    if is_training:
        # loading files
        plot_numpy_file_path = join(plot_folder, train_plot_numpy_file)
        training_loss   = np.load(join(plot_folder, training_history_numpy_file))
        validation_loss = np.load(join(plot_folder, validation_history_numpy_file))

        # making the figure
        image= np.load(plot_numpy_file_path)
        sample_fig = px.imshow(image, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)

        # creating the plot
        progress_plot = go.Figure()
        progress_plot.add_trace(go.Scatter(y=training_loss,
                    mode='lines',
                    name='Training Loss'))

        progress_plot.add_trace(go.Scatter( y=validation_loss,
                    mode='lines',
                    name='Validation Loss'))

        progress_plot.update_layout(
                            legend=dict(
                                x=0,
                                y=0,
                                traceorder="normal",
                                font=dict(
                                    family="sans-serif",
                                    size=12,
                                    color="black"
                                ),
                            )
                        )

        return sample_fig, {'visibility':'visible'},progress_plot,{'visibility':'visible'}
                
    else:
        return (dash.no_update,)*4


@app.callback(
    Output("epoch_number_status", 'children'),
    Input('epoch_number', 'value'))
def set_epoch_numbers(epoch_number):
    """
    Updates the number of epochs
    inptus:
        epoch_number (int): number of epochs to train the model
    outputs:
        epoch_number_status children (string): the status update 
    """

    global number_of_epochs
    if epoch_number is None:
        return (dash.no_update)
    else:
        number_of_epochs = epoch_number
        return ("Number of Epochs: {}".format(number_of_epochs))

@app.callback(
        Output("reset_model_status", 'children'),
        Input("reset_model", "n_clicks"))
def reset_model(n_clicks):
    """
    Creates the model at loading, then upon click, resets the model, and probably solve all human problems
    inputs:
        reset_model n_clicks (int): clicks on the button
    outputs:
        reset_model_status children (string): the status update 
    """
    global vae_model, options, optimizer

    if n_clicks is None:
        options = Option_list(n_class= 52)
        vae_model = VAE(options, device).to(device)
        optimizer = optim.Adam(vae_model.parameters(), lr = options.lr)
        return "Model initiated"
    else:
        print(torch.cuda.memory_allocated())
        vae_model = None
        print(torch.cuda.memory_allocated())
        vae_model = VAE(options, device).to(device)
        print(torch.cuda.memory_allocated())
        return "Model reset"

@app.callback(
        Output("save_model_status", 'children'),
        Input("save_model", "n_clicks"))
def save_model(n_clicks):
    """
    Saves the model and overrides the previous save. 
    inputs:
        save_model n_clicks (int): clicks on the button
    outputs:
        save_model_status children (string): the status update 
    """

    global vae_model
    if n_clicks is None:
        return "No model saved"
    else:
        model_save_path = join(model_folder, saved_model_file)
        torch.save(vae_model, model_save_path)
        return "Model saved"

@app.callback(
    Output('training_status', 'children'),
    Output('validation_status', 'children'),
    Input('training_model', 'n_clicks'))
def train_model_call(n_clicks):
    """
    Trains the model for a given number of epochs. 
    inputs:
        training_model n_clicks (int): clicks on the button
    outputs:
        save_model_status children (string): the status update 
    """
    global vae_model, train_dataset, test_dataset, train_iterator, test_iterator, options, optimizer
    global is_training

    if n_clicks is None:
        selected_label_path = join(data_folder, selected_labels_file)
        labels = np.load(selected_label_path)
        msg = "Preselected sample size: {}".format(labels.shape[0])
        return (dash.no_update, msg)
    else:
        is_training = True
        train_dataset, test_dataset, train_iterator, test_iterator = load_training_data()

        options.N_EPOCHS = number_of_epochs

        vae_model, train_loss_history, eval_loss_history = train_model(vae_model, optimizer, train_iterator, test_iterator, device, options)

        train_report = "Training Loss: {:2f}".format(train_loss_history[-1])
        eval_report = "Eval Loss: {:2f}".format(eval_loss_history[-1])
        
        is_training = False

        return (train_report, eval_report)

def load_training_data():
    """
    Loads the training data from the data folder
    """
    global raw_data, train_dataset, test_dataset, train_iterator, test_iterator, options
    
    x_path = join(data_folder, selected_data_file)
    y_path = join(data_folder, selected_labels_file)

    raw_data = Handwiriting_dataset(resize_input=False, 
                                x_path= x_path, 
                                y_path =y_path)
                                
    return dataloader_creator(raw_data, options)


##########################################################################################
###### Merging functions
##########################################################################################
@app.callback(
    Output('mrege_data_status', 'children'),
    Input('merge_data','n_clicks' ))
def merge_all_samples_in_data_folder(input_value):
    """
    Reads the data folder and finds all files that match specific pattern, then merge them in a 
    file, this file can be used in the next step to feed the t-sne algorithm.
    inputs: 
        merge_data n_clicks (int): checks if the button is pressed
    outputs:
        mrege_data_status children (string): updates the status text
    """
    global shared_dataset
    global shared_dataset_labels

    if input_value is None:
            return "Press to merge" 
    else:
        loaded_data_shape = merge_data()
        msg = "Merged and save data with shape {}".format(loaded_data_shape) 
        return msg

def merge_data():
    """
    A function to read all the alphabet_handwriting_64_init_n.py files
    and merge them in one single big file!
    """
    global  merged_dataset, num_of_classes, sample_size 

    # reading all the files in the folder
    all_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    data_files = [join(data_folder,f) for f in all_files if f[:24] == samples_file_name_pattern]
    
    merged_dataset = np.empty(shape = (1, 64,64))
    merged_dataset_label= np.empty(shape = (1, num_of_classes))

    for file in data_files:
        tmp_data = np.load(file)

        # make sure that the last blank cells are ommited
        if tmp_data.shape[0] > sample_size:
            tmp_data = tmp_data[:sample_size]
        if tmp_data.ndim > 3:
            tmp_data = tmp_data.reshape(-1, 64, 64)

        y = create_labels()
        merged_dataset = np.vstack((merged_dataset, tmp_data))
        merged_dataset_label = np.vstack((merged_dataset_label, y))

    merged_dataset = merged_dataset[1:]
    merged_dataset_label = merged_dataset_label[1:]

    # report
    print ("Size of merged data: {}".format(merged_dataset.shape))
    print ("Size of merged data labels: {}".format(merged_dataset_label.shape))
    
    # saving
    merged_dataset_file_path = join(data_folder, data_file_path)
    merged_dataset_label__file_path = join(data_folder, data_label_file_path)

    np.save(merged_dataset_file_path, merged_dataset)
    np.save(merged_dataset_label__file_path, merged_dataset_label)

    return merged_dataset.shape

def create_labels():
    """
    Convert the labels to one-hot vectors and saves them.
    """
    global num_of_classes,  sample_size, sample_size_each_char

    y = np.empty((0, num_of_classes))

    for i in range (num_of_classes):
        one_hot_vec = np.zeros((sample_size_each_char, num_of_classes))
        one_hot_vec[:,i] = 1
        y = np.vstack((y, one_hot_vec))

    return y

def read_merged_data():
    """
    Loads the data from the disk at any moment
    Used multiple times accross the code to make sure that the data is correct
    and of the same size everytime
    """
    global shared_dataset_labels
    global shared_dataset

    shared_dataset_path = join(data_folder, data_file_path)
    shared_dataset_labels_path = join(data_folder, data_label_file_path)

    shared_dataset = np.load(shared_dataset_path)
    shared_dataset_labels = np.load(shared_dataset_labels_path)

    print("-------------------------")
    print ("data loaded: {}".format(shared_dataset.shape))
    print("-------------------------")


##########################################################################################
###### Plot functions
##########################################################################################
@app.callback(
    Output('sample_size_status', 'children'),
    Output('sample_size', 'max'),
    Output('data_plot', 'figure'),
    Output('data_plot', 'style'),
    Output('data_plot_label', 'figure'),
    Output('data_plot_label', 'style'),
    Input('sample_size', 'value'))
def update_sample_size(sample_size):
    """
    Changes the samples to be displayed on the data curation plots
    """
    global tsne_samples_size
    global shared_dataset_labels
    
    if sample_size is None:
        return (dash.no_update,)* 6
    else:
        tsne_samples_size = sample_size
        scatter_plot, scatter_plot_label = plot_data_wrapper()
        max_value = shared_dataset_labels.shape[0]
        return ["Sample size: {}".format(sample_size), max_value,
                scatter_plot , {'visibility': 'visible'},
                scatter_plot_label , {'visibility': 'visible'}]

def plot_data_wrapper():
    """
    generates the two plots for the main data distribution plots
    """
    global shared_dataset_labels
    global shared_dataset
    global show_visualization
    global tsne_samples_size
    global tsne_data

    # handle the data loading and labling
    read_merged_data()

    show_visualization = True
    tsne_data_file_path = join(data_folder, tsne_distribution_numpy_file)
    tsne_data = np.load(tsne_data_file_path)[:tsne_samples_size]
    labels_as_number = np.array([np.where(r==1)[0][0] for r in shared_dataset_labels[:tsne_samples_size]])
    raw_data_embedded_df = pd.DataFrame({'x': tsne_data[:,0], 
                    'y': tsne_data[:,1],
                    'label': labels_as_number,
                    })
   
    # creating the plots
    # plot with tsne embedding on x and y axis
    scatter_plot = px.scatter(raw_data_embedded_df, x= "x",y="y",  color='label')
    scatter_plot.update_layout(coloraxis_showscale=False)
    scatter_plot.update_layout(dragmode="select")
    scatter_plot.update_xaxes(showticklabels=False)
    scatter_plot.update_yaxes(showticklabels=False)  
    scatter_plot.update_layout(clickmode='event+select')
    scatter_plot.update_traces(marker_size=4)

    # plot with tsne embedding on x and labels on y aixs
    scatter_plot_label = px.scatter(raw_data_embedded_df, x= "x",y='label',  color='label')
    scatter_plot_label.update_layout(coloraxis_showscale=False)
    scatter_plot_label.update_layout(dragmode="select")
    scatter_plot_label.update_xaxes(showticklabels=False)
    scatter_plot_label.update_yaxes(showticklabels=False)  
    scatter_plot_label.update_layout(clickmode='event+select')
    scatter_plot_label.update_traces(marker_size=4)
    return scatter_plot, scatter_plot_label

# global parameters to store the last state of the 
# hover data, helps with finding the active plot  
prev_hover_data_0 = None
prev_hover_data_1 = None

@app.callback(
    Output('hover_data_fig', 'figure'),
    Output('hover_data_fig', 'style'),
    Input('data_plot', 'hoverData'),
    Input('data_plot_label', 'hoverData'))
def display_hover_data(hoverData_0, hoverData_1):
    """
    A function to show the sampels while hovering the mouse cursor over
    the samples in any of the two main plots
    """
    global shared_dataset_labels
    global shared_dataset
    global prev_hover_data_0 
    global prev_hover_data_1
    
    sample_index = None

    # toggling between the hovering data and use the current one
    if hoverData_0 != None:
        if prev_hover_data_0 == None:
            prev_hover_data_0 = hoverData_0
            sample_index = hoverData_0["points"][0]['pointNumber']
        else:
            if prev_hover_data_0 != hoverData_0:
                sample_index = hoverData_0["points"][0]['pointNumber']
                prev_hover_data_0 = hoverData_0

    if hoverData_1 != None:
            if prev_hover_data_1 == None:
                prev_hover_data_1 = hoverData_1
                sample_index = hoverData_1["points"][0]['pointNumber']
            else:
                if prev_hover_data_1 != hoverData_1:
                    sample_index = hoverData_1["points"][0]['pointNumber']
                    prev_hover_data_1 = hoverData_1
    

    # Displaying the samples
    if show_visualization and sample_index != None:
        #images = shared_dataset[sample_index]
        #print (images.shape)
        images = making_grid_image(shared_dataset, sample_index)
        sample_fig = px.imshow(images, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)
        return [sample_fig,{'visibility': 'visible'}]
    else:
        return (dash.no_update,)* 2  

def making_grid_image(data, sample_index):
    """
    A utility function to build a grid of images from the 
    samples in the dataset that are before and after the
    hovered sample. Also handles cases of having a sample from the 
    the begining and the end of the list
    inputs 
        data (nparray): all samples
        sample_index (int): index of the item that the mouse hovers on
    returns:
        image (nparray): a 2d nparray of shape 64*items_to_show x 64*items_to_show
                         that has total_items cells of the same char
    """
    items_to_show = 3
    total_items = items_to_show**2

    # placeholder for these two values
    start_index = 0
    end_index = total_items

    # handling the edge cases:
    if 3 < sample_index%36 < 32:
        start_index = sample_index-(total_items//2)
        end_index = sample_index+(total_items//2)

    elif sample_index%36 <= 3:
        start_index = sample_index
        end_index = sample_index+total_items

    elif sample_index%36 >= 32:
        start_index = sample_index-total_items
        end_index = sample_index

    if items_to_show%2 !=0:
        end_index +=1 


    sample_indices = np.arange(start_index, end_index)   
    images = np.zeros(shape=(64*items_to_show, 64*items_to_show))
    
    for i in range (items_to_show):
        for j in range (items_to_show):
            id = i*items_to_show + j
            
            images[i*64:(i+1)*64,j*64:(j+1)*64] = data[sample_indices[id]]

    return images


##########################################################################################
###### selection functions
##########################################################################################
@app.callback(
    Output('selected_data_fig', 'figure'),
    Output('selected_data_fig', 'style'),
    Output('selected_indices', 'children'),
    Input('data_plot', 'selectedData'),
    Input('data_plot_label', 'selectedData'))
def show_selected_data(selectedData,selectedData_label):
    """
    Shows the samples that are selected by the user.
    inputs:
        selectedData: the indices of selected data
        selectedData_label: the lables of selected data [0, 51]
    outputs:
        selected_data_fig figure (px.imshow): grid image of selected samples
        selected_data_fig style (dic): style set to visible
        msg (string): update message
    """
    global shared_dataset_labels
    global shared_dataset

    data = []

    if selectedData != None:
        data.extend(selectedData["points"])
    if selectedData_label != None:
        data.extend(selectedData_label["points"])

    if show_visualization and len(data)> 0:
        selected_indices = [d['pointIndex'] for d in data]
        size_message = len(selected_indices)

        # saving the data on the disk
        path_to_save = "./data/"
        np.save(join(path_to_save,"selected_data"),shared_dataset[selected_indices])
        np.save(join(path_to_save,"selected_labels"), shared_dataset_labels[selected_indices])

        # making the plot
        images = making_grid_selected_image(shared_dataset,selected_indices )
        sample_fig = px.imshow(images, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)

        msg = "{} item(s) selected.".format(size_message)
        return [sample_fig,{'visibility': 'visible'}, msg]

    else:
       return (dash.no_update,)* 3 

def making_grid_selected_image(data, indices, items_to_show = 20):
    """
    Makes a grid of size NxN (N= items_to_show) from the samples 
    in the dataset which their indices are given.
    If the sample size is bigger than NxN, then it only shows NxN of them
    selected randomly.
    inputs:
        data (nparray): sample data
        indices (list): indices of samples to show
        items_to_show (int): number of items to be shown
    output:
        images (nparray): the grid of samples
    """

    total_items = items_to_show**2

    if len (indices) > total_items:
        indices = np.random.choice(indices, size = total_items, replace=False)
    images = np.ones(shape=(64*items_to_show, 64*items_to_show))

    for i in range (items_to_show):
        for j in range (items_to_show):
            id = i*items_to_show + j
            if id < len(indices):
                images[i*64:(i+1)*64,j*64:(j+1)*64] = data[indices[id]]
    return images


##########################################################################################
###### t-SNE functions
##########################################################################################
@app.callback(
    Output('tsne_button', 'children'),
    Output('tsne_status', 'children'),
    Output('loading_output', 'children'),
    Input('tsne_button', 'n_clicks'))
def tsne_processing(input_value):
    """
    Runs the t-SNE algorithm to distribute the high-dimensional pixel
    data into a 2D space. May take a few minutes to run. 
    """
    global shared_dataset_labels
    global shared_dataset
    global tsne_samples_size
    if input_value is None:
        raise PreventUpdate
    else:
        read_merged_data()

        if input_value is None:
            return ["t-SNE Process", "Press to perform t-SNE", ""]
        else:
            message = tsne_algorithm(shared_dataset)
            return  ["t-SNEd", message, ""]  

def tsne_algorithm(data):
    """
    Quick wrapper to run the openTSNE implementation of t-SNE algorithm
    This implementation only reads 1-D data, so it needs some preparations
    It saves the data on a numpy file.
    inputs:
        data (nparray): data in its original format (n, 64, 64)
    outputs:
        msg (string): updating message
    """
    tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=16,
            verbose= True,
            random_state=42,
        )

    raw_data = np.copy(data)
    raw_data = raw_data.reshape(raw_data.shape[0], -1)

    print ("start t-SNE")
    raw_data_embedded = tsne.fit(raw_data)
    print ("Finished t-SNE")
    
    tsne_data_path = join(data_folder, tsne_distribution_numpy_file)
    np.save(tsne_data_path, raw_data_embedded)

    msg = "processed t-SNE for {} data points".format(raw_data.shape[0])
    return (msg)


app.run_server(debug=True, port=8020)
# app.run_server(port=8020)