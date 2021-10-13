'''
    File name: dash_elements.py
    Author: Ardavan Bidgoli
    Date created: 10/13/2021
    Date last modified: 10/13/2021
    Python Version: 3.8.5
    License: MIT
'''
from dash import dcc, html 
import dash_bootstrap_components as dbc


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
                                html.H4("t-SNE", className="card-title"),
                                dbc.Button(id='merge_data', children= "Merge Data", color="dark", className="mr-1" ),
                                html.Br(),
                                dbc.Label(id='mrege_data_status', children="Data merge"),
                                html.Br(),
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