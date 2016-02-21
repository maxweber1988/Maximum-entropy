import numpy as np
import matplotlib.pyplot as plt

def calc_configuration(configuration,I):

	centers = configuration[0,:]
	widths = configuration[1, :]
	heights = configuration[2, :]

	result = np.zeros(centers[len(centers)-1]+widths[len(centers)-1])

	for i in range(len(centers)):
		result[centers[i]-widths[i]/2:centers[i]+widths[i]/2+1] = heights[i]

	return result

def update1(configuration,width_range):
	number_updates = np.random.randint(1,configuration.shape[1])
	print number_updates
	for i in range(number_updates):
		index = np.random.randint(0,configuration.shape[1])
		area = configuration[1,index] * configuration[2,index]
		configuration[1,index] += np.random.randint(- width_range,width_range)
		configuration[2,index] = area / configuration[1,index]
	return configuration

def update2(configuration,number_updates):
	area = np.sum(configuration[1,:] * configuration[2,:])
	choices = np.random.randint(0,2,number_updates)
	for i in range(len(choices)):
		choice = choices[i]
		if choice:
			index = np.random.randint(0,configuration.shape[1])
			configuration = np.delete(configuration, index, 1)
			new_area = np.sum(configuration[1,:] * configuration[2,:])
			configuration[2,:] *= new_area / area
			area = new_area
		else:
			index = np.random.randint(0,configuration.shape[1])

# centers = np.array([10,15,16,25])
# widths = np.array([6,10,4,6])
# heights = np.array([1.,2.,3.,1.])
centers = np.random.randint(10,200,200)
widths = np.random.randint(2,10,200)
heights = np.random.randint(1,20,200)
configuration = np.array([centers,widths,heights])
print configuration.shape
A = calc_configuration(configuration,5.)
plt.plot(A)
new_config = update1(configuration,4)
A_new = calc_configuration(new_config,1)
plt.plot(A_new)
plt.show()