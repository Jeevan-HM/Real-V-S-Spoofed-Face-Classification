import cv2 
import spoofed as sp

def image_capture():
    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()
        cv2.imshow("image", image)
        
        ## Stop Video Capture ##
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return image

image = image_capture()

path = "./resources/anti_spoof_models"
sp.test(image, path , 0)

## Test on a folder of images ##

# image = "./input/Fake_Faces/"
# i = 1
# for image_path in os.listdir(image):
#     # print(str(image_path))
#     image_path = os.path.join(image, image_path)
#     print(i)
#     path = "./resources/anti_spoof_models"
#     sp.test(image_path, path , 0)
#     i += 1