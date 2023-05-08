import cv2
import numpy as np 
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.models import load_model
import sys
# https://github.com/irdanish11/AnomalyEventDetection/blob/master/DeployModel.py



vc=cv2.VideoCapture('test.avi') #SET VID LOC
rval=True
print('Loading model')
MODEL_PATH = 'model.h5'
MODEL = load_model(MODEL_PATH)
print('Model loaded')

def mean_squared_loss(x1,x2):
	''' Compute Euclidean Distance Loss  between 
	input frame and the reconstructed frame'''
	diff=x1-x2
	a,b,c,d,e=diff.shape
	n_samples=a*b*c*d*e
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist


threshold=0.0004
counter=0
while (vc.isOpened()):
	imagedump=[]
	for i in range(20):
		rval,frame=vc.read()
		try:
			frame=resize(frame,(227,227,3))
		except:
			print("\n--------------Video ended--------------\n");exit();quit();sys.exit()
		#Convert the Image to Grayscale
		gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
		gray=(gray-gray.mean())/gray.std()

		gray=np.clip(gray,0,1)
		imagedump.append(gray)

	imagedump=np.array(imagedump)

	imagedump.resize(227,227,10)
	imagedump=np.expand_dims(imagedump,axis=0)
	imagedump=np.expand_dims(imagedump,axis=4)

	counter=counter+1
	print('Processing data for frame :',counter)

	output=MODEL.predict(imagedump)

	loss= mean_squared_loss(imagedump,output)
 
	if loss>threshold:
		print('Anomalies Detected in frame ',counter)
		#viewer=skimage.viewer.ImageViewer(imagedump)
		#viewer.show()
	else :
		print('No anomalies detected in frame',counter)
print('-------------------------------------------')

print("---------------------------------\nVideo complete\n")