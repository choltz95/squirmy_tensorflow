import numpy as np
import tensorflow as tf
import binascii

NUM_DIGITS = 180

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def binary_encode(s):
    bin_arr = [x[-3:] for x in [t for t in map(text_to_bits,s)]]
    bit_arr = []
    for binstr in bin_arr:
        bit_arr.extend(list(binstr))
    bit_arr = [int(i) for i in bit_arr]
    l = len(bit_arr)
    bit_arr.extend([0 for _ in range(NUM_DIGITS - l)])
    return np.array( bit_arr ) 

trX = np.load("trX.npy")
trY = np.load("trY.npy")

# Randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Our variables. The input has width NUM_DIGITS, and the output has width 1.
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 1])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 1])

# Predict y given x using the model.
py_x = model(X, w_h, w_o)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

BATCH_SIZE = 128

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(20000):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    test = ["ACATATAGACATACGT","AAA","TTT", "ATGC"]
    teX = np.array([binary_encode(x) for x in test])
   # teXX = np.transpose(teX)
    teY = sess.run(predict_op,feed_dict={X: teX})
    print(teY)

