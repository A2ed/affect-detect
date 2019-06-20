# Affect Detect 
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)


A framework for detecting 7 emotional primitives in facial expressions. The model is based on ResNet34 and trained using the FastAI library on the FER2013 dataset. It uses OpenCV to detect faces in a streaming video and calculates probabilistic estimates of emotional expressions using convolutional neural networks.

**Current Model Performance -  67.5% accuracy**

The 7 emotional primitives are:

- Happy
- Sad
- Neutral
- Fear
- Surprise
- Angry
- Disgust

## Getting Started

These instructions will get this model up and running. Follow them to make use of the `fertestcusstom.py` file to recognize facial emotions using custom images. This model can also be used as facial emotion recognition part of projects with broader applications

### Prerequisites
Install the packages in the requirements.txt to get up and running. Ideally you should do this in a virutal environment as the FastAI code base is frequently updated.

The main packages are:
- FastAI
- open-cv-contrib
- flask (if you want to run the web app)

### Quick Install

```
Clone the repository https://github.com/A2ed/affect-detect.git
```

After activating your virtual environment, install the packages in the requirements.txt using the following command.

```
pip install -r requirements.txt
```

Once installed, you can then run the app by using the following command in your terminal.

```
python3 scripts/video.py
```

This will use OpenCV to pop up a window with a stream from your webcam. For each frame, a face detection algorithm is run and the identified faces are run through the model to determine emotional expression. I currently have a probabilistic filter set on the model inference, so that it only outputs an identified emotion if inferred probability is above 40%. I found this to be a good balance point to get continuous readings, though you may want to play with it yourself.

**To quit the script, press q at any time.**

### Web App

Also included in the repo is a Flask web app that will perform the same thing as the video.py script. I'm currently developing this, but it's functional now in a minimal form. 

To get this running, first switch into the app/ subfolder.

```
cd path/to/app
```

You then need to set the environmental variable for Flask in your terminal session.

```
export FLASK_APP=web_app.py
```

Next, run the command below and copy the url into a browser.

```
flask run
```

To exit the webcam streaming, simply hit the escape key.

## Development Plans

For the web app, my next steps are to create a dashboard that displays the time course of emotions, as well as some summary metrics about the proportions of the different emotions captured.

For the model, I'll be refining it further by evaluating different face detection models and training it further using more varied emotional images than is contained in FER2013. 
