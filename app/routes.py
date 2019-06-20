from flask import render_template, Response
from app import app

from scripts.camera import *

def gen(camera):
	while True:
		em, frame = camera.decode()
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def home():
	# put code here
	return Response(gen(AffectDetect()), 
					mimetype='multipart/x-mixed-replace; boundary=frame')
	#return render_template('home.html', **params)


