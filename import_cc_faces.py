from PIL import Image
import numpy as np
import cv2
import time

def detect(filename, face_cascade=None):
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    return faces

def detect_faces(filename, im_size):
    faces_dict = {}
    
    faces = detect(filename)
    faces_list = [[int(f2) for f2 in f] for f in list(faces)]

    num_faces = len(faces)
        
    im_size = im_size[0] * im_size[1]
    
    if (num_faces) > 0:
        face_sizes = faces[:,2] * faces[:,3]
        total_face_percentage = float(np.sum(face_sizes)) / im_size
        largest_face_percentage = float(np.max(face_sizes)) / im_size
    else:
        face_sizes = []
        total_face_percentage = 0.0
        largest_face_percentage = 0.0
    
    faces_dict['list'] = faces_list
    faces_dict['num'] = num_faces
    faces_dict['largest'] = largest_face_percentage
    faces_dict['total'] = total_face_percentage
    
    return faces_dict