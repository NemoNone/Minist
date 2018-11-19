# from tensorflow.examples.tutorials.mnist import input_data
# minst = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
# print("Training data size:",minst.train.num_examples)
# print("Validating data size:", minst.validation.num_examples)
# print("Test data size:", minst.test.num_examples)
# print("Example training data size:", minst.train.images[0])
# print("Example training data label:", minst.train.labels[0])
import  tensorflow as  tf
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])
sess = tf.InteractiveSession()
print(tf.greater(v1,v2).eval())
print(tf.where(tf.greater(v1, v2), v1, v2).eval())