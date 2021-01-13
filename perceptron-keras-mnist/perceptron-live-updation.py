#################################### NOTEBOOK IMPORTS ######################################################
import numpy as np
import matplotlib.pyplot as plt

#################################### DRAWING THE LINE ######################################################
def draw(x1, x2, line_parameters, points, y):
    
    ln = plt.plot(x1, x2, '-')
    
    # converting matrix to array
    error = np.array(calculate_error(line_parameters, points, y))
    
    plt.title(f'Error: {error[0][0]:.4f}')
    
    plt.pause(0.0001)
    ln[0].remove()
    
#################################### SIGMOID ACTIVATION FUNCTION ###########################################
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

################################## CROSS ENTROPY ERROR CALCULATION #########################################
def calculate_error(line_parameters, points, y):
    
    # number of data points
    m = points.shape[0]
    
    # calculating probabilites
    p = sigmoid(points * line_parameters)
    
    ce = -(1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    
    return ce


##################################### GRADIENT DESCENT #####################################################
def gradient_descent(line_parameters, points, y, learning_rate):
    
    # number of data points
    m = points.shape[0]
        
    for i in range(1000):

        p = sigmoid(points * line_parameters)

        gradient = points.T * (p - y) * (learning_rate / m)
        
        # parameters updation
        line_parameters = line_parameters - gradient

        # getting new parameters (weights and bias)
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        # getting line coordinates to draw
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
        
        # drawing the line    
        draw(x1, x2, line_parameters, points, y)
        
        print(calculate_error(line_parameters, points, y))
        
     
    print("Final Error : ", calculate_error(line_parameters, points, y))   

        
        
####################################### DATA MAKING ######################################################
np.random.seed(42)

n_pts = 100

bias = np.ones(n_pts)


top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T

# Merging all points
all_points = np.vstack((top_region, bottom_region))

# Defining initial weights and bias
line_parameters = np.matrix([np.zeros(3)]).T

# Labels
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(2 * n_pts, 1)

# Figure Building
_, ax = plt.subplots(figsize=(4, 4))

ax.scatter(top_region[:,0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:, 1], color='b')

gradient_descent(line_parameters, all_points, y, 0.06)


plt.show()