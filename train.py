# kyle was here
import numpy as np
import gzip
import matplotlib.pyplot as plot

GUI = False

### Variables & Config ###
learning_rate = 0.2
epochs = 20
batch_size = 64
hidden_neurons = 128

### Helper Functions ###
def relu(x):
    return np.maximum(x, 0)
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0, keepdims=True)
def relu_deriv(x):
    return x > 0
def onehot(x):
    one_hot = np.zeros((x.size, x.max() + 1))
    one_hot[np.arange(x.size), x] = 1
    return one_hot.T

# Neural Network functions
def forward(input, w1, b1, w2, b2):
    hidden_output = w1.dot(input) + b1
    hidden_relu = relu(hidden_output)
    final_output = w2.dot(hidden_relu) + b2
    final_softmax = softmax(final_output)
    return hidden_output, hidden_relu, final_output, final_softmax

def backward(pixels, labels, layer1, layer1_relu, layer2_softmax, hidden_weights):
    batch_size = labels.size
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

def predict(X, w1, b1, w2, b2):
    _, _, _, a2 = forward(X, w1, b1, w2, b2)
    return np.argmax(a2, axis=0)

def test_accuracy(X, Y, w1, b1, w2, b2):
    prediction = predict(X, w1, b1, w2, b2)
    return np.sum(prediction == Y) / Y.size

### Train Model ###
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

if GUI:
    # setup plot
    print("Creating pyplot...")
    plot.ion()
    fig, ax = plot.subplots()
    line, = ax.plot([], [], "r-")
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    accuracies = []
    epoch_list = []

# Initialize weights
print(f"Initializing with {hidden_neurons} neurons...")
w1 = np.random.rand(hidden_neurons, 784) - 0.5
b1 = np.random.rand(hidden_neurons, 1) - 0.5
w2 = np.random.rand(10, hidden_neurons) - 0.5
b2 = np.random.rand(10, 1) - 0.5

# training loop
print("Beginning Training...")
images = x_train.shape[1]

for i in range(epochs):
    # shuffle images
    permutation = np.random.permutation(images)
    x_train_shuffled = x_train[:, permutation]
    y_train_shuffled = y_train[permutation]
    # mini batch loop
    for j in range(0, images, batch_size):
        # get batch
        begin = j
        end = j + batch_size
        x_batch = x_train_shuffled[:, begin:end]
        y_batch = y_train_shuffled[begin:end]
        # Train on this batch
        z1, a1, z2, a2 = forward(x_batch, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward(x_batch, y_batch, z1, a1, a2, w2)
        w1, b1, w2, b2 = update(w1, dw1, b1, db1, w2, dw2, b2, db2) 
    # check accuracy once per epoch on the whole dataset
    _, _, _, a2_full = forward(x_train, w1, b1, w2, b2)
    predictions = np.argmax(a2_full, axis=0)
    acc = np.sum(predictions == y_train) / y_train.size
    if GUI:
        # add plot info
        accuracies.append(acc)
        epoch_list.append(i)
        # add to plot
        line.set_data(epoch_list, accuracies)
        ax.relim()
        ax.autoscale_view()
        plot.draw()
        plot.pause(0.01)
if GUI:
    # finish plot drawing
    plot.ioff()
    plot.show()
# save model
print("Saving matrices...")
np.savez("model.npz", w1=w1, b1=b1, w2=w2, b2=b2)
# print final accuracy
acc = test_accuracy(x_test, y_test, w1, b1, w2, b2)
print(f"Final Test Accuracy: {acc * 100:.2f}%")