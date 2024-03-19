import numpy as np
import gzip

def get_data(path_images, path_labels, n_elements):
	with gzip.open(path_images, "rb") as fd1, gzip.open(path_labels, "rb") as fd2:
		fd1.read(16)
		fd2.read(8)
		images = np.split(np.array(list(fd1.read()))/255, n_elements)
		labels = [np.arange(10)==label for label in list(fd2.read())]
		return list(zip(images, labels))

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class Neural_network:
	def __init__(self, shape, is_random=False):
		self.shape = shape
		self.W = np.array([np.random.randn(y, x)/np.sqrt(x) if is_random else np.empty([y, x]) for x, y in zip([0] + shape[:-1], [0] + shape[1:])], dtype=object)
		self.b = np.array([np.random.randn(size_layer) if is_random else np.empty(size_layer) for size_layer in [0] + shape[1:]], dtype=object)
		self.a = np.array([np.empty(size_layer) for size_layer in shape], dtype=object)
	
	def feedforward(self, image):
		W, b, a = self.W, self.b, self.a
		a[0] = image
		for i in range(1, len(self.shape)):
			a[i] = sigmoid(np.dot(W[i], a[i-1]) + b[i])
			
		return a[-1]
	
	def backpropagation(self, y):
		W, b, a = self.W, self.b, self.a
		gradient = Neural_network(self.shape)
		
		for i in range(len(self.shape)-1, 0, -1):
			gradient.a[i] = (a[i] - y) / (a[i] * (1 - a[i]) + 0.00001) if i==len(self.shape)-1 else np.dot(W[i+1].T, gradient.b[i+1])
			gradient.b[i] = gradient.a[i] * a[i] * (1 - a[i])
			gradient.W[i] = np.outer(gradient.b[i], a[i-1])
			
		return gradient
	
	def gradient_descent(self, gradients, learning_rate):
		self.W -= sum([gradient.W for gradient in gradients]) * (learning_rate/len(gradients))
		self.b -= sum([gradient.b for gradient in gradients]) * (learning_rate/len(gradients))

def learn(training_data, test_data, shape, learning_rate, minibatch_size):
	neural_network = Neural_network(shape, True)
	minibatches = [training_data[i:i+minibatch_size] for i in range(0, len(training_data), minibatch_size)]
	epoch = 0
	
	while True:
		n_right_train = 0
		for minibatch in minibatches:
			gradients = []
			for image, label in minibatch:
				n_right_train += np.argmax(neural_network.feedforward(image))==np.argmax(label)
				gradients.append(neural_network.backpropagation(label))
			neural_network.gradient_descent(gradients, learning_rate)
		
		n_right_test = sum([np.argmax(neural_network.feedforward(image))==np.argmax(label) for image, label in test_data])
		
		epoch += 1
		print(f"Epoch {epoch}: {n_right_train}/{len(training_data)} {n_right_test}/{len(test_data)}")

training_data = get_data("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000)
test_data = get_data("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000)
learn(training_data, test_data, [28*28, 400, 10], 0.5, 10)
