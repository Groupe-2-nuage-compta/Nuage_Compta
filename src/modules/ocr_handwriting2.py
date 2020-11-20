import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np
from keras.models import load_model


def auto_canny(image, sigma=0.33):
    """Determine these lower and upper thresholds.

    Args:
        image (str): image
        sigma (float, optional): [description]. Defaults to 0.33.

    Returns:
        [type]: [description]
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


imagePath = 'C:/Users/utilisateur/Desktop/Projets/Nuage/data/images/formation.jpeg'
model = load_model(
    'C:/Users/utilisateur/Desktop/Projets/Nuage/src/model.h5')

# Load the image file
image = cv2.imread(imagePath)

# Convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The Canny edge detection algorithm can be broken down into 5 steps:
# Step 1: Noise Reduction
# Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter. We have already seen this in previous chapters.
blurred = cv2.medianBlur(gray, 5)
edged = auto_canny(blurred)

# RETR_EXTERNAL gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
# CHAIN_APPROX_SIMPLE  removes all redundant points and compresses the contour, thereby saving memory.
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

# loop over the contours
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = gray[y:y + h, x:x + w]
    thresh = cv2.threshold(roi, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # re-grab the image dimensions (now that its been resized)
    # and then determine how much we need to pad the width and
    # height such that our image will be 32x32
    (tH, tW) = thresh.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)

    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    padded = cv2.resize(padded, (28, 28))
    # padded = transform.resize(padded, (28, 28, 1))
    # prepare the padded image for classification via our
    # handwriting OCR model
    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)
    # update our list of characters that will be OCR'd
    chars.append((padded, (x, y, w, h)))

# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
# OCR the characters using our handwriting recognition model

preds = model.predict(chars)
# define the list of label names
# labelNames = "0123456789"
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
cv2.imshow("Image", image)
cv2.waitKey(0)
