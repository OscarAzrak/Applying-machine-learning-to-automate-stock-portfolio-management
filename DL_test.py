import tensorflow as tf
import tensorflow.compat.v1 as v1
train_in = [
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]]

train_out = [
    [0],
    [1],
    [1],
    [1]]

w = tf.Variable(tf.random_normal([3, 1], seed=15))
print(w)

x = v1.placeholder(tf.float32,[None,3])
y = v1.placeholder(tf.float32,[None,1])

print(x,y)

output = tf.nn.relu(tf.matmul(x, w))

print(output)