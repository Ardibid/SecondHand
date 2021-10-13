# SecondHand
A toolkit to interface SecondHand a C-VAE model for handwriting typeface generation. 

## Guides

### Running the app

To run the app, run this script from the same folder that the drawing_app.py exists and then access the app from your browser:

```
python pyArm.py -mode local 
```

![demo](/media/data_viewer.gif?raw=true)


There are three modes of running the app:

* **local**: runs on the local machine, you can access it on http://127.0.0.1:8050/
* **debug**: runs on the local machine, with debugging options. you can access it on http://127.0.0.1:8050/
* **remote**: runs on the local machine as a server, you can access it from other deives on the same network on http://0.0.0.0:8050/, replace the 0.0.0.0 part with your server machine's IP address, i.e., http://192.168.86.34:8050/ 
