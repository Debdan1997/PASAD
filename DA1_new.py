import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.linalg import svd
from scipy.linalg import norm

# Load data and separate the required sets
data = np.genfromtxt(r"data/1 - Scenario DA1/xmv10_359_data_1.csv",delimiter = ",")
data = data[:,14]
train_data = data[:500]
departure_data = data[500:4000]
test_data = data[4000:4800]
example_data = data


N = train_data.shape[0]
L = N//2
K = N-L+1

# Form trajectory matrix
trajectory_matrix = hankel(train_data[:L],train_data[L-1:])

# Calculate SVD
U,S,V = svd(trajectory_matrix,full_matrices= False)

# Set r
r = 1

# Extract first r significant components
u = U[:,:r]
ut = np.transpose(u)

# Calculate its centroid
mean = np.mean(trajectory_matrix,1)
#print(mean.shape)

# Project it
projected_mean = ut.dot(mean)
#print("mean",projected_mean)


differences = np.zeros(3500)
test_differences = np.zeros(800)
example_diff = np.zeros(4300)

# Find max departure in training phases' calculated Departure scores
max_departure = 0
for i in range(3500-L+1):
    curr_data = np.array(departure_data[i:i+L])
    #print(curr_data.shape,ut.shape)
    projected_val = ut.dot(curr_data)    
    differences[i]= norm(projected_val-projected_mean)
    differences[i] = differences[i]*differences[i]
    max_departure = max(max_departure,differences[i])
   
#print(max_departure)

# Calculate Departure Scores
for i in range(501,4800):
    curr_data = np.array(example_data[i-L:i])
    projected_val = ut.dot(curr_data)   
    difference = norm(projected_val-projected_mean)
    #example_diff[i-501]= (difference.dot(difference))
    example_diff[i-501]= difference*difference

     
        
# Plot the Time series
plt.figure(figsize = (7,2))
plt.xlim(0,4800)
plt.plot([x for x in range(0,500)],train_data,c='b')
plt.plot([x for x in range(500,4000)],departure_data,c='k')
plt.plot([x for x in range(4000,4800)],test_data,c='r')
plt.show()


# Plot the Departure Scores        
plt.figure(figsize = (7,2))
plt.xlim(0,4800)
x_down = [x for x in range(500,4800)]
plt.axhline(y=max_departure,color = 'r',linestyle = ':')
plt.plot(x_down,example_diff[:])
plt.show()