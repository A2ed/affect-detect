import cv2
from fastai import *
from fastai.vision import *
import PIL


def predict_array(x):
	"""A function to predict the face array using fastai model"""
	
	# convert from array to image
	img = PIL.Image.fromarray(gray, 'L')
	# convert to fastai Image object
	img = Image(pil2tensor(img, dtype=np.float32).div_(255))
	pred = learn.predict(img)
	# set probability threshold
	if pred[2][int(pred[1])] > .3:
		em = pred[0]

	return em

# load fast.ai learner object
learn = load_learner('../models')

# load classifier
casc_path = '../haar/haarcascade_frontalface_default.xml'
face_casc = cv2.CascadeClassifier(casc_path)

# get webcam
video_capture = cv2.VideoCapture(0)

while True:
	# capture frame-by-frame
	frame = video_capture.read()[1]
	# convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces
	faces = face_casc.detectMultiScale(gray,
									   scaleFactor=1.1,
									   minNeighbors=5,
									   minSize=(75, 75),
									   flags=cv2.CASCADE_SCALE_IMAGE
									   )

	for (x, y, w, h) in faces:
		# draw a rectangle around the faces
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# extract face
		gray = gray[y:y+h, x:x+w]
		
		# get emotion prediction
		try: 
			emotion = predict_array(gray)
			cv2.putText(frame, str(emotion), (int(x+w/3), y+h+25), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
			
		except: pass
		
	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()