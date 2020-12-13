
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import json
import numpy as np

# Load model from Json file
json_file = open("models/model.json", "r")
loaded_model = json_file.read()
json_file.close()

load_model = model_from_json(loaded_model)
load_model.load_weights("models/model.h5")

test_image = image.load_img("extracted.png", target_size=(64, 64, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = load_model.predict(test_image)
# train.class_indices
if result[0][0] == 1:
    print("Real Face")
else:
    print("Fake Face")
