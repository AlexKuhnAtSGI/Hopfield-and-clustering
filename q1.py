import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class Hopfield(object):
	def __init__(self, train_dataset=[], storkey=False):
		self.train_dataset = train_dataset
		self.num_training = len(self.train_dataset)
		self.num_neurons = len(self.train_dataset[0][0])

		if storkey:
			self.storkey()
		else:
			self.hebbian()

	def hebbian(self):
		self.W = np.zeros([self.num_neurons, self.num_neurons])
		for image_vector, _ in self.train_dataset:
			self.W += np.outer(image_vector, image_vector) / self.num_neurons
		np.fill_diagonal(self.W, 0)

	def storkey(self):
		self.W = np.zeros([self.num_neurons, self.num_neurons])

		for image_vector, _ in self.train_dataset:
			hebbian = np.outer(image_vector, image_vector)
			np.fill_diagonal(hebbian, 0)

			net = np.dot(self.W, image_vector)

			pre = np.outer(image_vector, net)
			post = np.outer(net, image_vector)

			self.W = np.add(self.W, np.divide(np.subtract(hebbian, np.add(pre, post)), self.num_neurons))

		np.fill_diagonal(self.W, 0)

	def activate(self, vector):
		changed = True

		while changed:
			changed = False
			indices = [i for i in range(0, len(vector))]
			np.random.shuffle(indices)

			new_vector = np.copy(vector)
			
			for i in range(0, len(vector)):
				neuron_index = indices.pop()
				s = np.dot(new_vector, self.W[neuron_index])

				new_vector[neuron_index] = 1 if s >= 0 else -1

				changed = (not vector[neuron_index] == new_vector[neuron_index]) or changed

			vector = new_vector

		return vector
		
	def predict(self, item, ones, fives):

		pattern = np.array(self.activate(item[0]))
		actual = item[1]

		min = 100000000000000.0
		
		for i in ones:
			dist = np.linalg.norm(pattern - i[0])
			if dist < min:
				min = dist
				pred = i[1]
			dist = np.linalg.norm(np.multiply(-1, pattern) - i[0])
			if dist < min:
				min = dist
				pred = i[1]
				
		for i in fives:
			dist = np.linalg.norm(pattern - i[0])
			if dist < min:
				min = dist
				pred = i[1]
				
			dist = np.linalg.norm(np.multiply(-1, pattern) - i[0])
			if dist < min:
				min = dist
				pred = i[1]

		return actual == pred



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

ones_train = []
fives_train = []

ones_test = []
fives_test = []

for i in range(len(x_train)):
	if y_train[i] == 1:
		ones_train.append(x_train[i].reshape([1, 784])[0])
	elif y_train[i] == 5:
		fives_train.append(x_train[i].reshape([1, 784])[0])
		
for i in range(len(x_test)):
	if y_test[i] == 1:
		ones_test.append(x_test[i].reshape([1, 784])[0])
	elif y_test[i] == 5:
		fives_test.append(x_test[i].reshape([1, 784])[0])

ones_train = [[1 if p > 0 else -1 for p in v] for v in ones_train]
ones_train = [(x, 1) for x in ones_train]
np.random.shuffle(ones_train)

fives_train = [[1 if p > 0 else -1 for p in v] for v in fives_train]
fives_train = [(x, 5) for x in fives_train]
np.random.shuffle(fives_train)

ones_test = [[1 if p > 0 else -1 for p in v] for v in ones_test]
ones_test = [(x, 1) for x in ones_test]
np.random.shuffle(ones_test)

fives_test = [[1 if p > 0 else -1 for p in v] for v in fives_test]
fives_test = [(x, 5) for x in fives_test]
np.random.shuffle(fives_test)



test_set = ones_test + fives_test
print("Test set is", len(ones_test)/len(test_set), "% ones")
print("Test set is", len(fives_test)/len(test_set), "% fives")
np.random.shuffle(test_set)

##NOTE: This program takes a very long time to run. I clocked it at about 40 minutes (20 minutes per learning rule) using these values.
##If testing on your own, you may want to consider reducing num_iters or num_units.

num_iters = 3
num_units = 15
x = list(range(2, 2*(num_units+1), 2))

y_hebb = np.zeros(num_units, np.float)
y_storkey = np.zeros(num_units, np.float)
#Using Hebb rule
for iter in range(num_iters):
	print("Hebb iteration", iter+1, "of", num_iters)
	np.random.shuffle(ones_train)
	np.random.shuffle(fives_train)
	
	for i in range(1, num_units+1, 1):
		train_set = ones_train[:i] + fives_train[:i]

		np.random.shuffle(train_set)

		clf_hebb = Hopfield(train_set)

		acc_hebb = 0.0
		for _, image in enumerate(test_set):
			norm = clf_hebb.predict(image, ones_train[:i], fives_train[:i])
			acc_hebb += norm
		y_hebb[i-1] += acc_hebb / (len(test_set) * num_iters)
plt.plot(x, y_hebb)
plt.xlabel("Number of images trained on")
plt.ylabel("Accuracy")
plt.title("Hopfield network with Hebbian learning rule")
plt.show()

#Using Storkey rule
for iter in range(num_iters):
	print("Storkey iteration", iter+1, "of", num_iters)
	np.random.shuffle(ones_train)
	np.random.shuffle(fives_train)
	for i in range(1, num_units+1, 1):
		#print("Training w/ Storkey rule", i * 2)
		train_set = ones_train[:i] + fives_train[:i]
		
		np.random.shuffle(train_set)
		
		clf_storkey = Hopfield(train_set, True)

		acc_storkey = 0.0
		for _, image in enumerate(test_set):
			norm = clf_storkey.predict(image, ones_train[:i], fives_train[:i])
			acc_storkey += norm
		y_storkey[i-1] += acc_storkey / (len(test_set) * num_iters)
		#print("storkey accuracy training samples", i * 2, "accuracy", (acc_storkey / len(test_set)))
plt.plot(x, y_storkey)
plt.xlabel("Number of images trained on")
plt.ylabel("Accuracy")
plt.title("Hopfield network with Storkey learning rule")
plt.show()