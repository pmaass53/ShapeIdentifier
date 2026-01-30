import numpy as np
import gzip
import matplotlib.pyplot as plot

### Helper Functions ###
def relu(x):
    # cap at 0 to prevent overflow and inaccurate numbers
    return np.maximum(x, 0)
def softmax(x):
    # Subtract max to prevent overflow (numerical stability improvement)
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0, keepdims=True)
# Neural Network Functions
def forward(input, w1, b1, w2, b2):
    hidden_output = w1.dot(input) + b1
    # ReLU activation: make sure the output is in the right range ("clamping")
    hidden_relu = relu(hidden_output)
    final_output = w2.dot(hidden_relu) + b2
    # Softmax activation: subtract to prevent overflow
    final_softmax = softmax(final_output)
    return hidden_output, hidden_relu, final_output, final_softmax

print("Loading Dataset...")
with gzip.open("./mnist/t10k-images-idx3-ubyte.gz", "rb") as timgidx3:
    data = np.frombuffer(timgidx3.read(), np.uint8, offset=16)
    x_test = data.reshape(-1, 784).astype(np.float32).T / 255.0
with gzip.open("./mnist/t10k-labels-idx1-ubyte.gz", "rb") as tlblidx1:
    y_test = np.frombuffer(tlblidx1.read(), np.uint8, offset=8)
print("Loading matrices...")
model = np.load("model.npz")
# Extract weights
w1 = model['w1']
b1 = model['b1']
w2 = model['w2']
b2 = model['b2']
# interactive testing
while True:
    # Pick a random index
    index = np.random.randint(0, x_test.shape[0])
    # Get the image and label
    current_image = x_test[index]
    actual_label = y_test[index]
    network_input = current_image.reshape(784, 1)
    # Make Prediction
    prediction_probs = forward(network_input, w1, b1, w2, b2)
    predicted_label = np.argmax(prediction_probs)
    confidence = prediction_probs[predicted_label][0] * 100
    # visualize
    fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(current_image.reshape(28, 28), cmap='gray')
    ax1.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
    ax1.axis('off')
    # bar graph
    classes = np.arange(10)
    # flatten to 1D array
    probs = prediction_probs.flatten()
    # Color the bars: Green if correct, Red if wrong
    bar_color = 'green' if predicted_label == actual_label else 'red'
    ax2.bar(classes, probs, color=bar_color)
    ax2.set_xticks(classes)
    ax2.set_title(f"Confidence: {confidence:.2f}%")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    plot.suptitle("Close this window to see the next image...", fontsize=14)
    plot.tight_layout()
    plot.show()
    # ask wether to show another
    cont = input("Show another? (y/n): ")
    if cont.lower() != 'y' and cont != '':
        break