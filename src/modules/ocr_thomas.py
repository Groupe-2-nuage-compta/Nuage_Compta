import cv2
from keras.models import load_model
import numpy as np


def resize_image(img, size=(20, 20)):

    h, w = img.shape[:2]

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    if dif > (size[0] + size[1]):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]

    return cv2.resize(mask, size, interpolation)


image = cv2.imread(
    "C:/Users/utilisateur/Desktop/Projets/Nuage/data/images/formation.jpeg")
model = load_model(
    'C:/Users/utilisateur/Desktop/Projets/Nuage/src/model.h5')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(
    "C:/Users/utilisateur/Desktop/Projets/Nuage/data/staging/gray.jpeg", gray)

# Permet de convertir l'image en blanc sur fond noir. Etape nécessaire pour la méthode findContours de OpenCV et par rapport à notre cible
_, threshold = cv2.threshold(
    gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
cv2.imwrite(
    "C:/Users/utilisateur/Desktop/Projets/Nuage/data/staging/threshold.jpeg", threshold)

# Copie de l'image sur laquelle on va travailler.
target_image = threshold.copy()

# Detection des contours. On demande de prendre que les points externes avec RETR_EXTERNAL et d'optimiser le nombre de points avec CHAIN_APPROX_SIMPLE
contours = cv2.findContours(
    threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

roi_index = 0

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w < 50 and h < 50:
        continue

    # Extraction de la lettre dans l'image
    roi = target_image[y:y+h, x:x+w]
    cv2.imwrite(
        'C:/Users/utilisateur/Desktop/Projets/Nuage/data/staging/extract/region_{}.jpeg'.format(roi_index), roi)

    # Resize de la lettre pour être en 20x20 en gardant le ratio
    r_img = resize_image(roi)
    cv2.imwrite(
        'C:/Users/utilisateur/Desktop/Projets/Nuage/data/staging/resize/region_resize_{}.jpeg'.format(roi_index), r_img)

    # Ajout du padding de 4
    p_img = cv2.copyMakeBorder(r_img.copy(), 4, 4, 4, 4, cv2.BORDER_CONSTANT)
    cv2.imwrite(
        'C:/Users/utilisateur/Desktop/Projets/Nuage/data/out/region_padded_{}.jpeg'.format(roi_index), p_img)

    roi_index += 1

    p_img = np.expand_dims(p_img, axis=-1)
    chars.append((p_img, (x, y, w, h)))

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
cv2.imshow('Img', image)
cv2.waitKey(0)
