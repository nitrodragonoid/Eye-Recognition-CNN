import cv2
import os
import csv
import numpy as np
import sys
import tensorflow as wtf

from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.25


def main():

    # Get image arrays and labels for all image files
    images, labels = load_data('train','Training_set.csv')

    # Split data into training and testing sets
    labels = wtf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to a file
    filename = 'Eyes_model.h5'
    model.save(filename)
    print(f"Model saved to {filename}.")


def load_data(data_dir,filename):

    # Open csv file and make a dictionary of image names and their lables
    with open(filename,'r') as f:
        reader = csv.reader(f)
        fields = next(reader)
        rows = {}
        for row in reader:
            rows[row[0]] = row[1]

    images = []
    lables = []
    for files in os.walk(data_dir):
        names = list(files)[2]

        for img in names :
            # Read the images and resize it to the given size
            image = cv2.imread(os.path.join(files[0], img))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            # Make a list of images and a list of lables
            images.append(image)
            lables.append(1 if rows[img] == 'male' else 0)  
    
    return(images,lables)

load_data('train','Training_set.csv')


def get_model():
    # Defining  convolutional neural network 
    model = wtf.keras.Sequential([

        wtf.keras.layers.Conv2D(16 ,(3,3) ,activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)),
        wtf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        wtf.keras.layers.Conv2D(32 ,(3,3) ,activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)),   
        wtf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        wtf.keras.layers.Conv2D(32 ,(3,3) ,activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)),
        wtf.keras.layers.Flatten(),
        wtf.keras.layers.Dense(200, activation = "relu"),
        wtf.keras.layers.Dropout(0.5),
        wtf.keras.layers.Dense(200, activation = "relu"),
        wtf.keras.layers.Dropout(0.5),
        wtf.keras.layers.Dense(2, activation = "softmax")
    ])
    # Compile and return the model 
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
   main()

def testing(modelname, folder):
    x = input('Press enter to start testing')

    # Load the saved model
    model = wtf.keras.models.load_model('Eyes_model.h5')

    # Go throught he testing set
    for name, blank, files in os.walk('test'):
        for img in files:
            
            #Read and resize the images from the testing set
            image = cv2.imread(os.path.join(name, img))
            image = cv2.resize(image, (30, 30))
            
            # Use model to classify the image as the highest probability
            classification = model.predict([np.array(image).reshape(1,30,30,3)]).argmax()
            
            # Print the corresponding clasification
            if classification == 1:
                print(img,'male')
            else:
                print(img,'female')

testing('Eyes_model.h5', 'test')
