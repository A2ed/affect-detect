import cv2
from fastai import *
from fastai.vision import *
import PIL




class AffectDetect():

	def __init__(self):
		# load fast.ai learner object
		self.learn = load_learner('./models')

		# load classifier
		casc_path = './haar/haarcascade_frontalface_default.xml'
		self.face_casc = cv2.CascadeClassifier(casc_path)

		# get webcam
		self.video_capture = cv2.VideoCapture(0)

	def __del__(self):
		"""Close up shop when everything is done"""

		self.video_capture.release()
		cv2.destroyAllWindows()

	def predict_array(self, img):
		"""A function to predict the face array using fastai model"""
		
		# convert from array to image
		img = PIL.Image.fromarray(img, 'L')
		# convert to fastai Image object
		img = Image(pil2tensor(img, dtype=np.float32).div_(255))
		pred = self.learn.predict(img)
		# set probability threshold
		if pred[2][int(pred[1])] > .3:
			em = pred[0]

		return em

	def decode(self):
		# capture frame-by-frame
		frame = self.video_capture.read()[1]
		# convert to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces
		faces = self.face_casc.detectMultiScale(gray,
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
				emotion = self.predict_array(gray)
				cv2.putText(frame, str(emotion), (int(x+w/3), y+h+25), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
				# Display the resulting frame
				ret, jpeg = cv2.imencode('.jpg', frame)
				
				return emotion, jpeg.tobytes()

			except: pass
			
		# return None for emotion if no faces detected
		ret, jpeg = cv2.imencode('.jpg', frame)
		return None, jpeg.tobytes()

	
