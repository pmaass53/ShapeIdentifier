import numpy as np
import gzip
import matplotlib.pyplot as plot
from PIL import Image

GUI = False

### Helper Functions ###
def relu(x):
    return np.maximum(x, 0)
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0, keepdims=True)

# Neural Network Functions
def forward(input, w1, b1, w2, b2):
    hidden_output = w1.dot(input) + b1
    hidden_relu = relu(hidden_output)
    final_output = w2.dot(hidden_relu) + b2
    final_softmax = softmax(final_output)
    return hidden_output, hidden_relu, final_output, final_softmax

print("Loading Dataset...")
with gzip.open("./mnist/t10k-images-idx3-ubyte.gz", "rb") as timgidx3:
    data = np.frombuffer(timgidx3.read(), np.uint8, offset=16)
    x_test = data.reshape(-1, 784).astype(np.float32) / 255.0

with gzip.open("./mnist/t10k-labels-idx1-ubyte.gz", "rb") as tlblidx1:
    y_test = np.frombuffer(tlblidx1.read(), np.uint8, offset=8)

print("Loading matrices...")
try:
    model = np.load("model.npz")
    w1 = model['w1']
    b1 = model['b1']
    w2 = model['w2']
    b2 = model['b2']
except FileNotFoundError:
    print("Error: model.npz not found!")
    exit()

# interactive testing
img = Image.open(input("Enter Image: "))
img.convert("L")
img = img.resize((28, 28))
# Reshape to (784, 1) for the matrix math
network_input = img.reshape(784, 1)
_, _, _, prediction_probs = forward(network_input, w1, b1, w2, b2)
predicted_label = np.argmax(prediction_probs)
confidence = prediction_probs[predicted_label][0] * 100
if GUI:
    # visualize
    fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f"Predicted: {predicted_label}")
    ax1.axis('off')
    # bar graph
    classes = np.arange(10)
    probs = prediction_probs.flatten()
    bar_color = 'green'
    ax2.bar(classes, probs, color=bar_color)
    ax2.set_xticks(classes)
    ax2.set_title(f"Confidence: {confidence:.2f}%")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    plot.suptitle("Close this window to see the next image...", fontsize=14)
    plot.tight_layout()
    plot.show()
else:
    print("Predicted: " + predicted_label + ", " + confidence + " Confident")