import numpy as np
import gzip
import matplotlib.pyplot as plot
##########################################################################################
# Input layer (784, the pixels of the input image)                                       #
# Hidden Layer (10)                                                                      #
# Output Layer (10, represents how much the input image matches each digit respectively) #
##########################################################################################

### Variables & Config ###
learning_rate = 0.2
epochs = 1000

### Helper Functions ###
def relu(x):
    # cap at 0 to prevent overflow and inaccurate numbers
    return np.maximum(x, 0)
def softmax(x):
    # Subtract max to prevent overflow (numerical stability improvement)
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0, keepdims=True)
def relu_deriv(x):
    # determine wether true or false
    return x > 0
def onehot(x):
    # used to classify/convert categorial data into numerical format
    # OneHot encoding: a blank list of zeroes, except for one "hot" number
    one_hot = np.zeros((x.size, x.max() + 1))
    one_hot[np.arange(x.size), x] = 1
    # transpose for correct ordering
    return one_hot.T
# Neural Network functions
def forward(input, w1, b1, w2, b2):
    hidden_output = w1.dot(input) + b1
    # ReLU activation: make sure the output is in the right range ("clamping")
    hidden_relu = relu(hidden_output)
    final_output = w2.dot(hidden_relu) + b2
    # Softmax activation: subtract to prevent overflow
    final_softmax = softmax(final_output)
    return hidden_output, hidden_relu, final_output, final_softmax
def backward(pixels, labels, layer1, layer1_relu, layer2_softmax, hidden_weights):
    batch_size = labels.size
    # calculate error of the final output layer vs expected output
    output_error = layer2_softmax - onehot(labels)
    gradient_w2 = (1 / batch_size) * output_error.dot(layer1_relu.T)
    gradient_b2 = (1 / batch_size) * np.sum(output_error, axis=1, keepdims=True)
    hidden_error = hidden_weights.T.dot(output_error) * relu_deriv(layer1)
    gradient_w1 = (1 / batch_size) * hidden_error.dot(pixels.T)
    gradient_b1 = (1 / batch_size) * np.sum(hidden_error, axis=1, keepdims=True)
    return gradient_w1, gradient_b1, gradient_w2, gradient_b2
def update(w1, gradient_w1, b1, gradient_b1, w2, gradient_w2, b2, gradient_b2):
    w1 -= learning_rate * gradient_w1
    b1 -= learning_rate * gradient_b1
    w2 -= learning_rate * gradient_w2
    b2 -= learning_rate * gradient_b2
    return w1, b1, w2, b2
# accuracy and testing functions
def predict(X, w1, b1, w2, b2):
    _, _, _, a2 = forward(X, w1, b1, w2, b2)
    return np.argmax(a2, axis=0)
def test_accuracy(X, Y, w1, b1, w2, b2):
    prediction = predict(X, w1, b1, w2, b2)
    return np.sum(prediction == Y) / Y.size

### Train Model ###
# Load data
print("Loading Dataset...")
with gzip.open("./mnist/train-images-idx3-ubyte.gz", "rb") as imgidx3:
    data = np.frombuffer(imgidx3.read(), np.uint8, offset=16)
    x_train = data.reshape(-1, 784).astype(np.float32).T / 255.0
with gzip.open("./mnist/train-labels-idx1-ubyte.gz", "rb") as lblidx1:
    y_train = np.frombuffer(lblidx1.read(), np.uint8, offset=8)
with gzip.open("./mnist/t10k-images-idx3-ubyte.gz", "rb") as timgidx3:
    data = np.frombuffer(timgidx3.read(), np.uint8, offset=16)
    x_test = data.reshape(-1, 784).astype(np.float32).T / 255.0
with gzip.open("./mnist/t10k-labels-idx1-ubyte.gz", "rb") as tlblidx1:
    y_test = np.frombuffer(tlblidx1.read(), np.uint8, offset=8)
# setup plot
print("Creating pyplot...")
plot.ion()
fig, ax = plot.subplots()
line, = ax.plot([], [], "r-")
ax.set_xlim(0, epochs)
ax.set_ylim(0, 1) # Accuracy is between 0 and 1
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
accuracies = []
epoch_list = []
# Initialize weights and biases with random numbers
w1 = np.random.rand(10, 784) - 0.5
b1 = np.random.rand(10, 1) - 0.5
w2 = np.random.rand(10, 10) - 0.5
b2 = np.random.rand(10, 1) - 0.5

# training loop
print("Beginning Training...")
for i in range(epochs):
    # 1. Forward Pass: Pass in the training images (x_train)
    z1, a1, z2, a2 = forward(x_train, w1, b1, w2, b2)
    # 2. Backward Pass: Pass in the intermediate values and the correct labels (y_train)
    dw1, db1, dw2, db2 = backward(x_train, y_train, z1, a1, a2, w2)
    # 3. Update: Apply the gradients to the weights and biases
    w1, b1, w2, b2 = update(w1, dw1, b1, db1, w2, dw2, b2, db2)
    # update pyplot info
    predictions = np.argmax(a2, axis=0)
    acc = np.sum(predictions == y_train) / y_train.size
    accuracies.append(acc)
    epoch_list.append(i)
    # update every 10 epochs to save time
    if i % 10 == 0:
        line.set_data(epoch_list, accuracies)
        ax.relim()
        ax.autoscale_view()
        plot.draw()
        # pause to let GUI refresh
        plot.pause(0.01)
# finish plot drawing
plot.ioff()
plot.show()
# save model when done
print("Saving matrices...")
np.savez("model.npz", w1=w1, b1=b1, w2=w2, b2=b2)
# print accuracy
acc = test_accuracy(x_test, y_test, w1, b1, w2, b2)
print(f"Final Test Accuracy: {acc * 100:.2f}%")