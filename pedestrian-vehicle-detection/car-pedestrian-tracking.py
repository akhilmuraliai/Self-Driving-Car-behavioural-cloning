# importing opencv library
import cv2

# importing image/video files
image_1 = 'test-images/road_1.jpg'
image_2 = 'test-images/road_2.jpg'

video_1 = 'test-videos/subject.mp4'
video_2 = 'test-videos/pedestrians.mp4'

# harrcascade files
car_file = 'xml/cars.xml' 
pedestrian_file = 'xml/pedestrians.xml'

# creating classifier
car_tracker = cv2.CascadeClassifier(car_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)

# video object
video = cv2.VideoCapture(video_1) 

while 1 == 1:

	# reading frame by frame
	ret, frame = video.read()

	# converting frame to grayscale
	if ret:
		grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		break

	# car and pedestrian detection
	cars = car_tracker.detectMultiScale(grayscale_frame) 
	pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

	# car boundary line drawing
	for (x, y, w, h) in cars:
		cv2.rectangle(frame, (x + 1, y + 2), (x + w - 1, y + h - 2), (255, 0, 0), 2)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

	# pedestrian boundary line drawing
	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

	# display the output
	cv2.imshow('Autonomous Driving', frame)

	# to quit the window
	if cv2.waitKey(1) == ord('q'):
		break

# free the memory
video.release()
cv2.destroyAllWindows()

print("Code Completed!")