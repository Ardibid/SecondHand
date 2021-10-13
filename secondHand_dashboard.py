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
                                    dbc.Button(id='load_model', children= "Load Trained Model", color="dark", className="mr-1" ),
                                    html.P(id="load_model_status", children=""),
                                    dbc.Button(id='load_favorite_model', children= "Load Favorite Model", color="dark", className="mr-1" ),
                                    html.P(id="load_favorite_model_status", children=""),
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
app.layout = dbc.Container([
                    html.H1("SecondHand Dashboard"),
                    html.Hr(),
                    data_graphs,
                    html.Hr(),
                    operation_tabs,
                    ])#,fluid=True)




##########################################################################################
###### Generate functions
##########################################################################################
@app.callback(
    Output('test_render','figure'),
    Output('test_render', 'style'),
    Output('squeeze_factor_status', 'children'),
    Output('line_padding_status', 'children'),
    Input('text_input', "value"),
    Input('squeeze_factor', 'value'),
    Input('line_padding', 'value'),)
def output_text(text_input, squeeze_factor, line_padding):
    global char_data
    images = render_text (text_input, char_data, squeeze_factor, line_padding)
    sample_fig = px.imshow(images, binary_string=True)
    sample_fig.update_layout(coloraxis_showscale=False)
    sample_fig.update_xaxes(showticklabels=False)
    sample_fig.update_yaxes(showticklabels=False)

    squeeze_factor_msg = ("Squeeze Factor: {}".format(squeeze_factor))
    line_padding_msg = ("Line padding: {}".format(line_padding))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    sample_fig.write_image("./renders/rendered_text_{}.jpeg".format(timestr))

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
    )
def generate_samples(char_index, mean, std):
    global char_data, vae_model, raw_data, device

    char = string.ascii_letters[int(char_index)]
  
    images, char_data, generated_ones, remaining = single_letter_test(char_index, mean, std, vae_model, raw_data, char_data, device, 4)
    sample_fig = px.imshow(images, binary_string=True)
    sample_fig.update_layout(coloraxis_showscale=False)
    sample_fig.update_xaxes(showticklabels=False)
    sample_fig.update_yaxes(showticklabels=False)

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
        Input("load_model", "n_clicks"))
def load_model(n_clicks):
    global vae_model
    if n_clicks is None:
        return (dash.no_update)
    else:
        load_training_data()
        vae_model = torch.load("./models/trained_model") 
        vae_model.eval()
        print (vae_model)
        return "Model loaded"


@app.callback(
        Output("load_favorite_model_status", 'children'),
        Input("load_favorite_model", "n_clicks"))
def load_model(n_clicks):
    global vae_model
    if n_clicks is None:
        return (dash.no_update)
    else:
        load_training_data()
        vae_model = torch.load("./models/favorite_model") 
        vae_model.eval()
        print (vae_model)
        return "Favorite Model loaded"



@app.callback(
        Output("save_typeface_status", 'children'),
        Output("latest_typeface",'figure'),
        Output('latest_typeface', 'style'),
        Input("save_typeface", "n_clicks"))
def save_typeface(n_clicks):
    global char_data
    if n_clicks is None:
        return (dash.no_update)
    else:
        images = finalize_font(char_data)
        sample_fig = px.imshow(images, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)


        timestr = time.strftime("%Y%m%d-%H%M%S")
        sample_fig.write_image("./renders/font_catalogue_{}.jpeg".format(timestr))
        
        return ("Typeface saved", 
                sample_fig, 
                {'visibility':'visible'})

##########################################################################################
###### Train functions
##########################################################################################
@app.callback(Output('train_progress', 'figure'),
              Output('train_progress', 'style'),
              Output('train_progress_graph', 'figure'),
              Output('train_progress_graph', 'style'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    global is_training
    
    if is_training:
        image= np.load("./plots/train_plot.npy")
        sample_fig = px.imshow(image, binary_string=True)
        sample_fig.update_layout(coloraxis_showscale=False)
        sample_fig.update_xaxes(showticklabels=False)
        sample_fig.update_yaxes(showticklabels=False)

        training_loss   = np.load("./plots/training_loss_history.npy")
        validation_loss = np.load("./plots/validation_loss_history.npy")

        progress_plot = go.Figure()

        progress_plot.add_trace(go.Scatter(y=training_loss,
                    mode='lines',
                    name='Training Loss'))
        
        progress_plot.add_trace(go.Scatter( y=validation_loss,
                    mode='lines',
                    name='Validation Loss'))

        progress_plot.update_yaxes(type="log")
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

        return sample_fig, {'visibility':'visible'},progress_plot, {'visibility':'visible'}
    else:
        return (dash.no_update,)*4


@app.callback(
    Output("epoch_number_status", 'children'),
    Input('epoch_number', 'value'))
def set_epoch_numbers(epoch_number):
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
    global vae_model, options, optimizer
    if n_clicks is None:
        options = Option_list(n_class= 52)
        vae_model = VAE(options, device).to(device)
        optimizer = optim.Adam(vae_model.parameters(), lr = options.lr)
        return "Model created"
    else:
        vae_model = VAE(options, device).to(device)
        return "Model reset"

@app.callback(
        Output("save_model_status", 'children'),
        Input("save_model", "n_clicks"))
def save_model(n_clicks):
    global vae_model
    if n_clicks is None:
        return "No model saved"
    else:
        torch.save(vae_model,"./models/trained_model")
        return "Model saved"

@app.callback(
    Output('training_status', 'children'),
    Output('validation_status', 'children'),
    Input('training_model', 'n_clicks'))
def train_model_call(n_clicks):
    global vae_model, train_dataset, test_dataset, train_iterator, test_iterator, options, optimizer
    global is_training
    if n_clicks is None:
        labels = np.load("./data/selected_labels.npy")
        return (dash.no_update, "Preselected sample size: {}".format(labels.shape[0]))
    else:
        is_training = True
        train_dataset, test_dataset, train_iterator, test_iterator = load_training_data()

        options.N_EPOCHS = number_of_epochs

        vae_model, train_loss_history, eval_loss_history = train_model(vae_model, optimizer, train_iterator, test_iterator, device, options)

        #reconstruct_figure = batch_recon(vae_model, options,train_iterator)

        train_report = "Training Loss: {:2f}".format(train_loss_history[-1])
        eval_report = "Eval Loss: {:2f}".format(eval_loss_history[-1])
        
        is_training = False
        return (train_report,eval_report)

def load_training_data():
    global raw_data, train_dataset, test_dataset, train_iterator, test_iterator, options

    raw_data = Handwiriting_dataset(resize_input=False, 
                                x_path= "./data/selected_data.npy", 
                                y_path = "./data/selected_labels.npy",)
                                
    return dataloader_creator(raw_data, options)


##########################################################################################
###### Merging functions
##########################################################################################
@app.callback(
    Output('mrege_data_status', 'children'),
    Input('merge_data','n_clicks' ))
def merge_data_from_server(input_value):
  global shared_dataset
  global shared_dataset_labels
  if input_value is None:
        return "Press to merge" 
  else:
    loaded_data_shape = merge_data()
    return "Merged and save data with shape {}".format(loaded_data_shape) 

def merge_data():
    """
    A function to read all the alphabet_handwriting_64_init_n.py files
    and merge them in one single big file!
    """
    global  merged_dataset
    num_of_classes= 52
    path_to_search = "./data/"

    # reading all the files in the folder
    all_files = [f for f in listdir(path_to_search) if isfile(join(path_to_search, f))]
    data_files = [join(path_to_search,f) for f in all_files if f[:24] == "alphabet_handwriting_64_"]
    print (data_files)
    merged_dataset = np.empty(shape = (1, 64,64))
    merged_dataset_label= np.empty(shape = (1, num_of_classes))

    for file in data_files:
        tmp_data = np.load(file)

        # make sure that the last blank cells are ommited
        if tmp_data.shape[0] > 1872:
            tmp_data = tmp_data[:1872]
        if tmp_data.ndim > 3:
            tmp_data = tmp_data.reshape(-1, 64, 64)

        y = create_labels(tmp_data.shape[0], num_of_classes)
        merged_dataset = np.vstack((merged_dataset, tmp_data))
        merged_dataset_label = np.vstack((merged_dataset_label, y))

    merged_dataset = merged_dataset[1:]
    merged_dataset_label = merged_dataset_label[1:]

    # report
    print ("Size of merged data: {}".format(merged_dataset.shape))
    print ("Size of merged data labels: {}".format(merged_dataset_label.shape))
    
    # saving
    np.save("./data/alphabet_handwriting_64", merged_dataset)
    np.save("./data/labels", merged_dataset_label)

    return merged_dataset.shape

def create_labels(sample_size = 1872, num_of_classes= 52):
    """
    Convert the labels to one-hot vectors and saves them.
    IMPORTANT NORE: IT ONLY WORKS IF YOU HAVE ALL 11 PAGES SCANNED CORRECTLY!
    """
    y = np.empty((0, num_of_classes))

    for i in range (num_of_classes):
        one_hot_vec = np.zeros((36, num_of_classes))
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
    shared_dataset_labels = np.load("./data/labels.npy")
    shared_dataset = np.load("./data/alphabet_handwriting_64.npy")
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

    """
    global shared_dataset_labels
    global shared_dataset
    global show_visualization
    global tsne_samples_size
    global tsne_data
    # handle the data loading and labling
    read_merged_data()
    show_visualization = True
    tsne_data = np.load('./data/tsne_data.npy')[:tsne_samples_size]
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
    hovered sample.
    """
    items_to_show = 3
    total_items = items_to_show**2

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
def show_selected_data(selectedData,selectedData_label ):
    """
    Shows the samples that are selected by the user.
    selectedData: the indices of selected data
    selectedData_label: the lables of selected data [0, 51]
    """
    global shared_dataset_labels
    global shared_dataset

    data = []

    if selectedData != None:
        data.extend(selectedData["points"])
    if selectedData_label != None:
        data.extend(selectedData_label["points"])

    if show_visualization and len(data)> 0:
        selected_indices = [datum['pointIndex'] for datum in data]
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
        return [sample_fig,{'visibility': 'visible'},("{} item(s) selected.".format(size_message))]

    else:
       return (dash.no_update,)* 3 

def making_grid_selected_image(data, indices, items_to_show = 20):
    """
    Makes a grid of size NxN (N= items_to_show) from the samples 
    in the dataset which their indices are given.
    If the sample size is bigger than NxN, then it only shows NxN of them
    selected random;y.
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
    np.save("./data/tsne_data", raw_data_embedded)
    return ("processed t-SNE for {} data points".format(raw_data.shape[0]))


app.run_server(debug=True, port=8020)
# app.run_server(port=8020)