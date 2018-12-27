# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os, glob
import numpy as np
import tensorflow as tf
import subprocess
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from imutils.video import VideoStream
from imutils.video import FPS

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))
#camera.start_preview()

modelFullPath = '/home/pi/Desktop/output/output_graph.pb'         # the path to the training model
labelsFullPath = '/home/pi/Desktop/output/output_labels.txt'      # the path to the text file that contains labels


# the function that creates the audio file (notices the user of the obstacle)
def say(words):
    tempfile="temp.wav"
    devnull=open("/dev/null","w")
    subprocess.call(["pico2wave", "-w", tempfile, words],stderr=devnull)
    subprocess.call(["aplay",tempfile],stderr=devnull)
    os.remove(tempfile)


# read transferred trained model
def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# distinguish whether the tactile pavement block is linear/circular in real-time
def run_inference_on_image():
    answer = None
    camera.capture("/home/pi/Desktop/temp_img.jpg")
    camera.start_preview()
    imagePath = "/home/pi/Desktop/temp_img.jpg"
    
    # if obstacles are not detected
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    create_graph()
    
    # if an obstacle is detected, recognize it
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        
        # sort prediction rates for each label from highest to lowest to recognize an obstacle
        top_k = predictions.argsort()[-1:][::-1]
        
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            #print('%s (score = %.5f)' % (human_string, score))
        
        # obstacle recognized... aurally notifies the user
        answer = labels[top_k[0]]
        if "stair" in human_string:
            say("There are stairs nearby.")
        elif "elevator" in human_string:
            say("There is an elevator.")
        elif "crosswalk" in human_string:
            say("There is a crosswalk ahead")
        elif "escalator" in human_string:
            say("There is an escalator ahead")

if __name__ == "__main__":
    run_inference_on_image()
