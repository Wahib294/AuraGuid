import threading
#from pynput import keyboard
import keyboard
import face_recognition
from imutils import paths
import pickle
import cv2
import os
import pyttsx3
import time


released = False
# Function to be executed in the first thread
video_capture = cv2.VideoCapture("http://192.168.1.23:81/stream")
def speak(name):
    text = " "
    if len(name) > 1:
        for i,j in enumerate(name):
            if i == len(name)-1:
                text += " " + j
            else:
                text +=  j + " and"
    else:
        text += name[0]
    print(text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()
def my_function():
    #find path of xml file containing haarcascade file 
    fflag = False
    cascPathface = "./haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    filename = "./face_enc"
    global video_capture
    global released
    while True:
        if not os.path.isfile(filename):
            flag = False
            imagePaths = list(paths.list_images('./Images'))
            knownEncodings = []
            knownNames = []
            # loop over the image paths
            for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                # load the input image and convert it from BGR (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #Use Face_recognition to locate faces
                boxes = face_recognition.face_locations(rgb,model='hog')
                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)
                # loop over the encodings
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    name = imagePath.split("/")
                    name = name[-1]
                    name = name.replace(".png","")
                    print(name)
                    knownNames.append(name)
            #save emcodings along with their names in dictionary data
            data = {"encodings": knownEncodings, "names": knownNames}
            #use pickle to save data into a file for later use
            f = open("face_enc", "wb")
            f.write(pickle.dumps(data))
            f.close()      
        data = pickle.loads(open('face_enc', "rb").read())
        print("Streaming started")
        while not exit_flag.is_set():
                    # grab the frame from the threaded video stream
            if released:
                exit()
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(60, 60),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            # convert the input frame from BGR to RGB
            # convert the input frame from BGR to RGB 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # the facial embeddings for face in input
            encodings = face_recognition.face_encodings(rgb)
            names = []
            # loop over the facial embeddings incase
            # we have multiple embeddings for multiple fcaes
            for encoding in encodings:
            #Compare encodings with encodings in data["encodings"]
            #Matches contain array with boolean values and True for the embeddings it matches closely
            #and False for rest
                matches = face_recognition.compare_faces(data["encodings"],
                encoding)
                #set name =inknown if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    #Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        #Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                        #increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                    #set name which has highest count
                    name = max(counts, key=counts.get)
                    name = os.path.basename(name)
                # update the list of names
                names.append(name)
                # loop over the recognized faces
                print(names)
            if len(names) > 0:
                threading.Thread(target=speak, args=(names,)).start()
                time.sleep(2)
        exit_flag.clear()

        #Recording Facial Features

        x = 0
        while x < 40:
            ret,frame = video_capture.read()
            x += 1 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame,model='hog')
        if len(boxes) == 0:
            print("No face detected")
            released = True
            video_capture.release()    
            exit()
        usr = input("Enter Name : ")
        cv2.imwrite(f'./Images/{usr}.png',frame)
        flag = False
        imagePaths = list(paths.list_images('./Images'))
        knownEncodings = []
        knownNames = []
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(rgb,model='hog')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
                name = imagePath.split("//")
                name = name[-1]
                name = name.replace(".png","")
                print(name)
                knownNames.append(name)
        #save emcodings along with their names in dictionary data
        data = {"encodings": knownEncodings, "names": knownNames}
        #use pickle to save data into a file for later use
        f = open("face_enc", "wb")
        f.write(pickle.dumps(data))
        f.close()       

# Function to be executed in the second thread (monitors key press)
#def on_key_release(key):
#    try:
#        k = key.char
#    except:
#        k = key.name

#    if k == target_key:
#        print("Target key pressed. Terminating thread 1...")
#        exit_flag.set()
#    elif key == keyboard.Key.esc:
#        video_capture.release()
#        cv2.destroyAllWindows()
#        exit()
# Set the target key and exit flag
target_key = "right"  # Change this to the desired key
exit_flag = threading.Event()

# Create and start the first thread
thread1 = threading.Thread(target=my_function)
thread1.start()

# Create and start the second thread (key listener)
#with keyboard.Listener(on_release=on_key_release) as listener:
#    listener.join()
while True: 
    if keyboard.read_key() == '1':    
        exit_flag.set()    
    elif keyboard.read_key() == '2':    
        released = True
        video_capture.release()    
        exit()
# Wait for both threads to finish
thread1.join()
