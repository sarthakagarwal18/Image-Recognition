import cv2
import sys
import create_csv
import pandas as pd
import numpy as np

if len(sys.argv)<2:
	print ("Please add Test Image Path")
	sys.exit()

test_img = sys.argv[1]

faceCascade = cv2.CascadeClassifier('haarcascade_face.xml')

def train():
	
	# Create 'train_faces.csv', which contains the images and their corresponding labels
	create_csv.create()
	
	# Face Recognizer using Linear Binary Pattern Histogram Algo
	face_recognizer = cv2.face.createLBPHFaceRecognizer()
	
	# Read csv file using pandas
	data = pd.read_csv('train_faces.csv').values
	
	images=[]
	labels=[]
	
	for ix in range(data.shape[0]):
		
		img = cv2.imread(data[ix][0])
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		images.append(gray)
		labels.append(data[ix][1])
	
	face_recognizer.train(images,np.array(labels))
	return face_recognizer
	
	
def test(test_img, face_recognizer):
	
	image = cv2.imread(test_img)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = 0)
	
	for (x, y, w, h) in faces:
		
		sub_img = gray[y:y+h,x:x+w]
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		# Predict label of detected face
		pred_label = face_recognizer.predict(gray)
		
		cv2.putText(image,str(pred_label),(x,y-5), cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0),1)
		cv2.imshow('Face Recognition',image)
		# Press Esc to Close Window
		cv2.waitKey(0)
	

if __name__ == '__main__':
	face_recog = train()
	test(test_img, face_recog)