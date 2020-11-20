import os
from PIL import Image
from IPython.display import SVG
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path 
import pytesseract
import cv2
import imutils
from imutils.contours import sort_contours
import livelossplot
plot_losses = livelossplot.PlotLossesKeras()


random.seed(80)


def create_train_data(DATADIR, IMG_SIZE, numbers=False):

    categories = ["A", "B", "C", "D", "E", "F", "G", "H",
                  "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                  "T", "Y", "V", "W", "X", "Y", "Z"]

    train_data = []
    X = []
    y = []

    # Recuperations des fichiers images et conversion
    i = 0
    for category in categories:
        path = os.path.join(DATADIR, category)
        class_num = categories.index(category)
        file_list = os.listdir(path)
        random.shuffle(file_list)
        for file in file_list:
            file_array = cv2.imread(os.path.join(
                path, file), cv2.IMREAD_GRAYSCALE)
            file_array = cv2.resize(file_array, (IMG_SIZE, IMG_SIZE))
            train_data.append([file_array, class_num])
            i += 1
            if i % 50000 == 0:
                print(f"Chargement {i} fichiers lettres")

    # Si l'utilisateur choisit de créer un modèle avec les chiffres, les données de mnist seront ajoutées au modèle
    i = 0
    if numbers == True:
        (X_mnist, y_mnist), _ = load_data()
        for number in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            categories.append(number)
        number_list = []
        for i, image in enumerate(X_mnist):
            index_target = categories.index(str(y_mnist[i]))
            train_data.append([image, index_target])
            i += 1
            if i % 5000 == 0:
                print(f"Chargement {i} fichiers chiffres")

    # Traitement final du dataset
    random.shuffle(train_data)
    for feature, label in train_data:
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0
    y = np.array(y)
    y = to_categorical(y)

    return X, y, categories


def create_and_train_model(X, y, outfile, categories):
    model = Sequential()

    model.add(Conv2D(128, (5, 5), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Activation("relu"))

    model.add(Dense(len(categories)))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])

    model.fit(X, y, batch_size=256, epochs=10,
              validation_split=0.25, callbacks=[plot_losses])

    model.save(f'./data/OUTPUT/{outfile}')
    model.summary()

    print("Model complete.")


def picture_analysis(image_path, model_path):
    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    model = load_model(model_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(gray, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 55 and h <= 120):
            h = h + 4
            w = w + 4
            y = y - 4
            x = x - 4
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=28)
            # otherwise, resize along the height
            elif tH > tW:
                thresh = imutils.resize(thresh, height=28)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=4, bottom=4,
                                        left=4, right=4, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))
            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

            #     # show the image
            # cv2.imshow("Image", padded)
            # cv2.waitKey(0)

    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    # OCR the characters using our handwriting recognition model

    preds = model.predict(chars)
    # define the list of label names
    labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    labelNames = [l for l in labelNames]

    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        # draw the prediction on the image
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # show the image
    plt.imshow(image)


def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

def convert_and_read_pdf(PDF_file, tesseract_path):
    # Store all the pages of the PDF in a variable 
    pages = convert_from_path(PDF_file, 500) 
    
    # Counter to store images of each page of PDF to image 
    image_counter = 1
    
    # Iterate through all the pages stored above 
    for page in pages: 
        # Declaring filename for each page of PDF as JPG 
        # For each page, filename will be: 
        # PDF page 1 -> page_1.jpg 
        # PDF page 2 -> page_2.jpg 
        # PDF page 3 -> page_3.jpg 
        # .... 
        # PDF page n -> page_n.jpg 
        filename = "./data/images/page_"+str(image_counter)+".jpg"
        
        # Save the image of the page in system 
        page.save(filename, 'JPEG') 
    
        # Increment the counter to update filename 
        image_counter = image_counter + 1
    
        # Variable to get count of total number of pages 
    filelimit = image_counter-1
    
    # Creating a text file to write the output 
    outfile = "./data/images/out_text.txt"
    
    # Open the file in append mode so that  
    # All contents of all images are added to the same file 
    f = open(outfile, "w") 

    pytesseract.pytesseract.tesseract_cmd = tesseract_path  
    # Iterate from 1 to total number of pages 
    for i in range(1, filelimit + 1): 
    
        # Set filename to recognize text from 
        # Again, these files will be: 
        # page_1.jpg 
        # page_2.jpg 
        # .... 
        # page_n.jpg 
        filename = "./data/images/page_"+str(i)+".jpg"
            
        # Recognize the text as string in image using pytesseract
        text = str(((pytesseract.image_to_string(Image.open(filename))))) 
    
        # The recognized text is stored in variable text 
        # Any string processing may be applied on text 
        # Here, basic formatting has been done: 
        # In many PDFs, at line ending, if a word can't 
        # be written fully, a 'hyphen' is added. 
        # The rest of the word is written in the next line 
        # Eg: This is a sample text this word here GeeksF- 
        # orGeeks is half on first line, remaining on next. 
        # To remove this, we replace every '-\n' to ''. 
        text = text.replace('-\n', '')     
    
        # Finally, write the processed text to the file. 
        f.write(text) 

        print(text)
    
    # Close the file after writing all the text. 
    f.close()