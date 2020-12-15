# Real v/s Fake/Spoffed Face Detection
### The aim of this project is to differentiate between a Real and Fake Face.
* ### _Real Face_ :
  Real Faces are those where an actual person is detected and is not a part of an inanimated object such as a smartphone.
* ### _Fake/Spoofed Face_ :
  Fake Face/Spoffed are Photos of a person shown from inanimated objects like Phone, ID Card etc. shown to the Camera.

#### _NOTE_ : This project is based on Python3.8
## Installing the requirements
```bash
pip -m install -r requirements.txt
```
## OR
```bash
pip3 install -r requirements.txt
```
### Running code
```bash
python3 face_detection.py
```
### Input for Program
- After runing the above command, webcam will turn on to take a picture of your face. To capture the picture press "q", program will capture the frame as input and predict if detected face category (Real/Spoofed)!
