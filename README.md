# Real v/s Fake/Spoffed Face Detection
### The aim of this project is to differentiate between a Real and Fake Face.
* ### _Real Face_ :
  Real Faces are those where an actual person is detected and is not a part of an inanimated object such as a smartphone.
* ### _Fake/Spoofed Face_ :
  Fake/Spoffed Face are the photos of a person shown from inanimated objects like a Phone, ID Card etc. shown to the Camera.

#### _NOTE_ : This project is built on Python3.8
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
- After runing the above command, the webcam will turn on. To capture the picture press "q", program will capture the frame as the input and predict if the detected face is  Real/Spoofed.

## Reference 
- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)