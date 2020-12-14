import dlib
import cv2 
import imutils 
import numpy as np
from keras.models import load_model, model_from_json
from keras.preprocessing import image as Image
from PIL import Image as im 
import cv2
import spoof as sp
import os

eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
glass_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye_tree_eyeglasses.xml")

# Fancy box drawing function by Dan Masek
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
 
    # Top left drawing
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right drawing
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left drawing
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right drawing
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
 
def brightness_correction(image):
    filtered_image = np.zeros(image.shape, image.dtype)
    alpha = 1
    beta = 50
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                filtered_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    return filtered_image

def linear_stretching(input, lower_stretch_from, upper_stretch_from):
    lower_stretch_to = 0  
    upper_stretch_to = 255
    output = (input - lower_stretch_from) * ((upper_stretch_to - lower_stretch_to) / (upper_stretch_from - lower_stretch_from)) + lower_stretch_to
    return output

def gamma_correction(image):
    gamma_image = image.copy()
    max_value = np.max(gamma_image)
    min_value = np.min(gamma_image)
    for y in range(len(gamma_image)):
        for x in range(len(gamma_image[y])):
            gamma_image[y][x] = linear_stretching(gamma_image[y][x], min_value, max_value)
    return gamma_image

def normalize_image(image):
    image = image.copy()
    output = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.equalizeHist(output)
    image = dst.copy()
    return image

def detect_glass(image):
    try:
        glass_image = image.copy()
        glass_rect = glass_cascade.detectMultiScale(glass_image, scaleFactor=1.2)
        # print("Glass Detection = " + str(len(glass_rect)))
        return {"detection": len(glass_rect)}
    except Exception as e:
        return{"detection": e}

def detect_eyes(image):
    try:
        eye_image = image.copy()
        eye_rect = eye_cascade.detectMultiScale(eye_image, scaleFactor=1.2, minNeighbors=5)
        eye_cords = []
        eye_detection = 0
        for (x, y, w, h) in eye_rect:
            if len(eye_rect) == 2:
                cv2.rectangle(eye_image, (x, y), (x + w, y + h), (255, 255, 255), 10)
                eye_cords.append([x,y,w,h])
        if len(eye_cords) >= 1:
            eye_detection = len(eye_cords)
        return {"detection": eye_detection}
    except Exception as e:
        return{"detection": e}
    
def cnn_model(image):
    try:
        img = im.fromarray(image) 
        json_file = open("models/model.json", "r")
        loaded_model = json_file.read()
        json_file.close()

        load_model = model_from_json(loaded_model)
        load_model.load_weights("models/model.h5")

        # try:
        test_image = Image.load_img('filtered.png', target_size=(64, 64, 3))
        test_image = Image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = load_model.predict(test_image)
        # train.class_indices
        print("Prediction = " + str(result[0][0]))
        if result[0][0] > 0.05:
            return {"message": "Real Face"}
        else:
            return {"message":"Fake Face"}
        # except:
        #     return{"message": "Fake Face"}
    except Exception as e:
        return {"message":e}
    
def vidoCapture():
    # Grab video from your webcam
    stream = cv2.VideoCapture(0)
    
    # Face detector
    detector = dlib.get_frontal_face_detector()

    while True:
        # read frames from live web cam stream
        (grabbed, frame) = stream.read()
    
        # resize the frames to be smaller and switch to gray scale
        frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Make copies of the frame for transparency processing
        overlay = frame.copy()
        output = frame.copy()
    
        # set transparency value
        alpha  = 0.5
    
        # detect faces in the gray scale frame
        face_rects = detector(gray, 0)
    
        # print("Before:" + str(len(face_rects)))
        # loop over the face detections
        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            # 313 154 493 334
            # draw a fancy border around the faces
            # if face_rects == 1:
            
            if (x1 > 200 and y1 > 140 and x2 > 400 and y2 > 300):
                # print(x1,y1,x2,y2)
                # print("After:" + str(len(face_rects) + 1))
                cv2.imwrite("extracted" + '.png', overlay)
                draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0), 2, 10, 10)
                cv2.putText(overlay, "face", (x2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (192,168,0), 2, cv2.LINE_AA)
        # make semi-transparent bounding box
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # show the frame
        cv2.imshow("Face Detection", output)
        key = cv2.waitKey(1) 
    
        # press q to break out of the loop
        if key == ord("q"):
            break
    
    # cleanup
    # stream.stop()
    stream.release()

    cv2.destroyAllWindows()

# try:
# vidoCapture()
# try:
#     image = cv2.imread("extracted.png")
#     original_height, original_width = image.shape[:2]
#     image_resize = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
#     brightness_corrected = brightness_correction(image_resize)
#     filtered_image = gamma_correction(brightness_corrected)
#     glasses = detect_glass(brightness_corrected)["detection"]
#     eyes = detect_eyes(image)["detection"]
#     print("Glasses: " + str(glasses)) 
#     print("Eyes: "  + str(eyes))
#     if glasses < 3:
#         filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
#         normalized_image = normalize_image(filtered_image)
#         # normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
#         # image_resize = cv2.resize(brightness_corrected, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
#         cv2.imwrite("filtered" + '.png', normalized_image)
#         print(cnn_model(image)["message"])
#         # print("Hello")
#         # os.remove("extracted.png")
#         # os.remove("filtered.png")
#     else:
#         print("Eyes not detected")
 
# create a new cam object
cap = cv2.VideoCapture(0)
# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
while True:
    # read the image from the cam
    _, image = cap.read()
    # converting to grayscale
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect all the faces in the image
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        print(x,y,width, height)
        if (x > 250 and y > 175 and width > 180 and height > 180):
            draw_border(image, (x, y), (x + width, y + height), (162, 255, 0), 2, 10, 10)
            cv2.putText(image, "face", (width, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (192,168,0), 2, cv2.LINE_AA)
            # cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
            cv2.imwrite("extracted" + '.png', image)
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

# except Exception as e:
#     print("Face Not Detected")
    
path = "./resources/anti_spoof_models"
sp.test("extracted.png", path , 0)
os.remove("extracted.png")
