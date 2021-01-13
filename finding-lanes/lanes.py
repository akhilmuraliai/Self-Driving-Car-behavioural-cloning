################################# Notebook Imports ###############################
import cv2
import numpy as np


############################### Canny Edge Detector ###############################
def canny(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0)
    canny = cv2.Canny(blurred, 50, 150)
    
    return canny


####################################### ROI #######################################
def region_of_interest(image):
    
    height = image.shape[0]
    triangle_polygon = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle_polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image


############################ Lines to Displayed on Image ############################
def display_lines(image, lines):
    
    lined_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lined_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    return lined_image


################################# Coordinate Making #################################
def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except:
        slope, intercept = 0.001, 0
        
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])


############################### Average Slope Intercept ###############################
def average_slope_intercept(image, lines):
    
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
    
    for line in lines:
        
        x1, y1, x2, y2 = line.reshape(4)
        
        # fitting polynomial
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    # taking average        
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    
    
    return np.array([left_line, right_line])



############################### Video Opening Part ################################
cap = cv2.VideoCapture('road.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    _, frame = cap.read()
    
    canny_image = canny(frame)
    
    cropped_image = region_of_interest(canny_image)
    
    lines = cv2.HoughLinesP(
        cropped_image, 2, 
        np.pi/180, 100, np.array([]),
        minLineLength=40, maxLineGap=5
    )
    
    average_lines = average_slope_intercept(frame, lines)
    
    avg_line_image = display_lines(frame, average_lines)
    
    lane_identified_image = cv2.addWeighted(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.8,
        avg_line_image, 1, 1
    )
    
    cv2.imshow('The-Window', cv2.cvtColor(lane_identified_image, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()