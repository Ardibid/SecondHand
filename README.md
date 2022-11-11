# SecondHand
A toolkit to interface SecondHand a C-VAE model for handwriting typeface generation.<br>

<img src="media\hero.gif" width="750"/>

--- 

## Guides

### Running the app

To run the app, run this script from the same folder that the secondHand_dashboard.py exists and then access the app from your browser:

```
python secondHand_dashboard.py -mode local 
```



There are three modes of running the app:

* **local**: runs on the local machine, you can access it on http://127.0.0.1:8050/
* **debug**: runs on the local machine, with debugging options. you can access it on http://127.0.0.1:8050/
* **remote**: runs on the local machine as a server, you can access it from other deives on the same network on http://0.0.0.0:8050/, replace the 0.0.0.0 part with your server machine's IP address, i.e., http://192.168.86.34:8050/ 

### Setup
* Clone the repository
* Make sure you have all the dependencies with the correct version, you can use the requirements.txt file
* Only for the first run:   
    * Run the app for the first time and use Merge Data button to merge all the sample data files
    * Run the t-SNE algorithm to create the t-SNE distribution
* Follow the three steps:
    * Create your data selection
    * Train your model
    * Generate type faces


## Data curation
Use the two scatter plots to select a portion of data set, or even the whole. The larger the selection, the longer it takes to train the model. You can inspect the samples using plots in the seconf row. You can use rectangle selection tool, lasso selection tool, and use shift to combine different selections. <br>
You can use the sample size slider to view more data at a time. By default you will see a small portion of data, to make it easier for your browser, RAM, and GPU. But you can cranck that slider to show you all +30K samples, your call!

<img src="media\data_viewer_lowres.gif" width="500"/>

## Model Training
Once you have your data selected, you can use the ``Train`` tab to train the C-VAE model. You can add/remove/edit your selection **after each round of training is over**. Your model will continue training with the new selection. 
You can save the model or simply reset it to start from the begining. Your model will be re-initiated, but your selection will stays the same.

<!-- ![demo](media\training_lowres.gif?raw=true) -->
<img src="media\training_lowres.gif" width="500"/>

## Typeface Generation
You can use the ``Generation`` tab to play with the model's latent space and fine tune each glyph in your tyepface.

<!-- ![demo](media\generation_short_cropped.gif?raw=true) -->
<img src="media\generation_short_cropped.gif" width="500"/>

Once Done with all the glyphs, you can save the typeface and start rendering text with it!

<!-- ![demo](media\render_cropped.gif?raw=true) -->
<img src="media\render_cropped.gif" width="500"/>

## Technicals


### Dependencies
(the toolkit is developed with the versions mentioned here, use other versions at your own risk)


* [dash](https://pypi.org/project/dash/): 2.2.0
* [dash_bootstrap_components](https://dash-bootstrap-components.opensource.faculty.ai/): 1.0.3
* [plotly](https://pypi.org/project/plotly/): 5.6.0
* [torch](https://pytorch.org/): 1.10.2+cu102
* [numpy](https://numpy.org/): 1.22.2
* [pandas](https://pandas.pydata.org/): 1.4.1
* [scipy](https://www.scipy.org/): 1.8.0
* [openTSNE](https://opentsne.readthedocs.io/en/latest/): 0.6.1
* [opencv-python](https://pypi.org/project/opencv-python/):4.5.5  
* [kaleido](https://pypi.org/project/kaleido/): 0.2.1

--- 
By [Ardavan Bidgoli](https://wwww.ardavan.io), Spring 2022, 
[GitHub](https://github.com/Ardibid) <br>
Developed as a part of Ardavan Bidgoli's PhD thesis and 48-770 @ Carnegie Mellon University, School of Architecture. <br>


