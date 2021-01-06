# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.util import deprecation
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from sklearn.cluster import KMeans

deprecation._PRINT_DEPRECATION_WARNINGS = False

class KMeansInitializer(initializers.Initializer):
	#Not entirely sure how Keras initializers work, but in this instance I require 2, one for betas and one for my cluster centres
	#I wanted to do the computing here but I need access to the cluster centers here and all the cluster points in the beta initializer
	#so the simplest way is to run kmeans externally and use the values it gives me in these 2 very basic initializers
	def __init__(self, centroids):
		super().__init__()
		self.centroids = centroids

	def __call__(self, shape, dtype=None):
		return self.centroids

# class BetasInitializer(initializers.Initializer):
	# def __init__(self):
		# super().__init__()

	# def __call__(self, km):
		# betas = 1.0
		# return betas
		
class RBFLayer(layers.Layer):
	def __init__(self, output_dim, initializer, betas_initializer=None, **kwargs):

		self.output_dim = output_dim
		self.initializer = initializer
		self.betas_initializer = initializers.Constant(1.0)
		super().__init__(**kwargs)
		
	def build(self, input_shape):
		self.centers = self.add_weight(name='centers',
									   shape=(self.output_dim, input_shape[1]),
									   initializer=self.initializer,
									   trainable=True)
		self.betas = self.add_weight(name='betas',
									 shape=(self.output_dim,),
									 initializer=self.betas_initializer,
									 trainable=True)

		super().build(input_shape)
		
	def call(self, x):
		mu = tf.expand_dims(self.centers, -1)
		return tf.exp(-self.betas * tf.math.reduce_sum(tf.transpose(mu - tf.transpose(x))**2, axis=1))
		
def optimal_K(X):
	#Running this function took about 2 hours to complete
	#I suggest changing the value of iterations below if you want to test that it works
	inertias=[]
	k_vals = list(range(10,115,5))
	
	iterations = 10
	
	for k in k_vals:
		print("Trying", k)
		km = KMeans(n_clusters=k, n_init=iterations)
		km.fit(X)
		inertias.append(km.inertia_)
	
	plt.plot(k_vals,inertias,'ro')
	plt.plot(k_vals,inertias, 'r')
	plt.xlabel("Number of Clusters")
	plt.ylabel("Within-Cluster Sum of Squares")
	plt.title("Optimal K-value for Clustering")
	plt.show()
		
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

xTrn = []
xTes = []

for x in x_train:
	xTrn.append(x.reshape(784)/255)
	
for x in x_test:
	xTes.append(x.reshape(784)/255)

#Uncomment the below to validate part 1: I suggest changing some of my values (iterations, k_vals) for testing purposes
#optimal_K(xTrn)

xTrn = np.array(xTrn)
xTes = np.array(xTes)

km = KMeans(n_clusters=50, n_init=5)
km.fit(xTrn)

model = keras.Sequential()
model.add(RBFLayer(50,KMeansInitializer(km.cluster_centers_)))
model.add(layers.Dense(1))
model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.02),
						  loss='mse',
						  metrics=['accuracy'])
model.fit(xTrn, y_train, epochs=20, batch_size=1000)
print(model.evaluate(x=xTes, y=y_test))
