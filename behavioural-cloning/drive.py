from flask import Flask
import socketio
import eventlet
from keras.models import load_model
import base64
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from io import BytesIO
# sockets are used to perform real tme communication between a client and a web Server
# when clients connects to a socket, it keeps listening for new events from Server
# allows to continously update client with data
# Bidirectional Connection

sio = socketio.Server()

app = Flask(__name__)


# when user visits /home the message is print
# @app.route('/home')
# def greetings():
# 	return 'Message From Flask: Never Stop!'

# if __name__ == '__main__':
# 	app.run(port=4000)
# localhost:4000/home in our browser


# preprocessing as we done in model training
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# listen to updates that will be sent to telemetry from the simulator
# as soon as connected, the steer and throttle are set to zero
# then simulator will send us back teh data which contains current image of the frame
# where the car is currently located in the track
# based on that image we run it through our model
# then the model will predict the steering_angle and we return it back to the simulation

speed_limit = 10

@sio.on('telemetry')
def telemetry(sid, data):
	speed = float(data['speed'])
	image = Image.open(BytesIO(base64.b64decode(data['image'])))
	image = np.asarray(image)
	image = img_preprocess(image) # at this moment our model is 3D
	image = np.array([image]) # model expects as 4D
	steering_angle = float(model.predict(image))
	throttle = 1.0 - speed/speed_limit
	print('st:{} th:{} sp:{}'.format(steering_angle, throttle, speed))
	send_control(steering_angle, throttle)

# fires upon a connection
# session-id, environment
@sio.on('connect')
def connect(sid, environ):
	print('Connected To Simulator (UDACITY)')
	send_control(0, 0)




def send_control(steering_angle, throttle):
	sio.emit('steer', data = {
		'steering_angle': steering_angle.__str__(),
		'throttle': throttle.__str__()
		})




if __name__ == '__main__':

	model = load_model('mymodel.h5')

	# combine socketio server with flask app
	app = socketio.Middleware(sio, app)
	# gateway server, IP and Port, empty IP - can listen on any IP addresses
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
