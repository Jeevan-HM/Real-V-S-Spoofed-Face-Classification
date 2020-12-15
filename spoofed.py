import os
import numpy as np
import cv2
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def test(image_name, model_dir, device_id):
    ## If input is an image from a folder ##
    # image_name = cv2.imread(image_name)
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image_name)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image_name,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1 and value >= 0.6:
        print("Real Face.")
        result_text = "RealFace"
        color = (255, 0, 0)
    else:
        print("Fake Face.")
        result_text = "FakeFace"
        color = (0, 0, 255)
    print("Prediction speed {:.2f} s".format(test_speed))
    
    
    