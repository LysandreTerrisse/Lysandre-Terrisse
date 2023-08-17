import numpy as np
import gzip

def get_data(path_images, path_labels, n_elements):
	with gzip.open(path_images, "rb") as fd1, gzip.open(path_labels, "rb") as fd2:
		fd1.read(16)
		fd2.read(8)
		images = [np.array(list(fd1.read(28*28))) for _ in range(n_elements)]
		labels = list(fd2.read(n_elements))
		return list(zip(images, labels))
			
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class Neural_network:
	def __init__(self, shape, is_random=False):
		self.shape = shape
		self.W = np.array([np.random.randn(y, x) if is_random else np.empty([y, x]) for x, y in zip([0] + shape[:-1], [0] + shape[1:])])
		self.b = np.array([np.random.randn(size_layer) if is_random else np.empty(size_layer) for size_layer in [0] + shape[1:]])
		self.a = np.array([np.empty(size_layer) for size_layer in shape])
	
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
			gradient.a[i] = 2 * (a[-1] - y) if i==len(self.shape)-1 else np.dot(W[i+1].T, gradient.b[i+1])
			gradient.b[i] = gradient.a[i] * a[i] * (1 - a[i])
			gradient.W[i] = np.outer(gradient.b[i], a[i-1])
			
		return gradient
	
	def gradient_descent(self, gradients, learning_rate):
		self.W += sum([gradient.W for gradient in gradients]) * (learning_rate/-len(gradients))
		self.b += sum([gradient.b for gradient in gradients]) * (learning_rate/-len(gradients))
	
def learn(training_data, test_data, shape, learning_rate, minibatch_size):
	neural_network = Neural_network(shape, True)
	minibatches = [training_data[i:i+minibatch_size] for i in range(0, len(training_data), minibatch_size)]
	epoch = 0
	
	while True:
		n_right_train, n_right_test = 0, 0
		for minibatch in minibatches:
			gradients = []
			for image, label in minibatch:
				predicted_number = np.argmax(neural_network.feedforward(image))
				n_right_train += predicted_number==label
				y = np.arange(shape[-1])==label
				gradients.append(neural_network.backpropagation(y))
			neural_network.gradient_descent(gradients, learning_rate)
			
		for image, label in test_data:
			predicted_number = np.argmax(neural_network.feedforward(image))
			n_right_test += predicted_number==label
		
		epoch += 1
		print(f"Epoch {epoch}: {n_right_train}/{len(training_data)} {n_right_test}/{len(test_data)}")

training_data = get_data("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000)
test_data = get_data("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000)
learn(training_data, test_data, [28*28, 16, 16, 10], 0.01, 10)