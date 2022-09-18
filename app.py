import flask
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
# from flask_wtf import FlaskForm
# from wtforms import FileField, SubmitField
from tensorflow import keras
# from flask.ext.wtf import Form
# from werkzeug import secure_filename
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd() , "images")


@app.route("/" , methods=["GET","POST"])
def home():
    return render_template("homepage.html")

with open("mappings_18092022211.json" , 'r') as f:
    jsonFile = json.load(f)

FinalImages = []

@app.route("/result" , methods=["POST" ])
def result():

    try:

        if request.method== "POST":
            file = request.files["file1"]
            if file == "":
                return "No file Selected"
            global imageName
            imageName = str(secure_filename("image." + str(file.filename.split(".")[-1] )))
            file.save(imageName) # file.filename
        print(file)

        newImg = preprocess()
        cls = get_predictions(newImg)
        if get_class_name(cls, jsonFile) == "Not Found":
            message = "There are no images of the given class in our dataset."
            return render_template("Warning.html" , message = message)

        else:
            global FinalImages
            getImages( get_class_name(cls, jsonFile) )
            print("2",FinalImages)

            print(str(FinalImages[0]).replace("\\", "//"))

            return render_template("Images.html" , a =str("static/"+FinalImages[0]), b= "static/"+FinalImages[1] , c= "static/"+FinalImages[2] , d= "static/"+FinalImages[3] , e= "static/"+FinalImages[4] , f= "static/"+FinalImages[5])
            # return flask.send_file("FinalImages[0]" , mimetype='image/gif')

    except ValueError:
        return render_template("Warning.html" , message = "No File Selected")



def preprocess():
    global imageName
    img = np.expand_dims(resize(imread(imageName),(299,299,3)), axis = 0)
    return img

def get_predictions(img):
    preds = model.predict(img)
    cls = np.argmax(preds)
    return cls


model = keras.applications.InceptionResNetV2(include_top = True,weights = "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5")

def get_class_name(cls_id, new_mappings):
    possible = []
    possible_count = []
    for i in new_mappings:
        if str(cls_id) in new_mappings[i].keys():
            possible.append(i)
            possible_count.append(new_mappings[i][str(cls_id)])
    if len(possible)==0 :


        return "Not Found"
    return possible[possible.index(max(possible))]

def getImages(name):
    for j in os.listdir("101_ObjectCategories"):
        if name == j:
            break
    count = 0
    global FinalImages
    FinalImages = []
    for i in os.listdir(os.path.join("101_ObjectCategories", j)):
        FinalImages.append(os.path.join(os.path.join("101_ObjectCategories" , j), i))
        count += 1
        if count >7:
            break
    print("1:",FinalImages)

app.run()
