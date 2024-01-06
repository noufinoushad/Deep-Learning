MLP for Binary Classification
  use the Ionosphere binary (two-class) classification dataset to demonstrate an MLP for binary classification.

This dataset involves predicting whether a structure is in the atmosphere or not, given radar returns.
The dataset will be downloaded automatically using Pandas, but you can learn more about it here.
Ionosphere Dataset (csv).
Ionosphere Dataset Description (csv).
You can use a LabelEncoder to encode the string labels to integer values 0 and 1. The model will be fit on 67% of the data, and the remaining 33% will be used for evaluation, split using the train_test_split() function.
It is good practice to use ‘relu‘ activation with a ‘he_normal‘ weight initialization. This combination goes a long way in overcoming the problem of vanishing gradients when training deep neural network models.
The model predicts the probability of class 1 and uses the sigmoid activation function. The model is optimized using the adam version of stochastic gradient descent and seeks to minimize the cross-entropy loss.


MLP for Multiclass Classification

We will use the Iris flowers multiclass classification dataset to demonstrate an MLP for multiclass classification.
This problem involves predicting the species of iris flower given measures of the flower.
The dataset will be downloaded automatically using Pandas, but you can learn more about it here.
Iris Dataset (csv).
Iris Dataset Description (csv).
Given that it is a multiclass classification, the model must have one node for each class in the output layer and use the softmax activation function. The loss function is the ‘sparse_categorical_crossentropy‘, which is appropriate for integer encoded class labels (e.g., 0 for one class, 1 for the next class, etc.)



MLP for Regression


Use the Boston housing regression dataset to demonstrate an MLP for regression predictive modeling.
This problem involves predicting house value based on the properties of the house and neighborhood.
The dataset will be downloaded automatically using Pandas, but you can learn more about it here.
Boston Housing Dataset (csv).
Boston Housing Dataset Description (csv).
This is a regression problem that involves predicting a single numerical value. As such, the output layer has a single node and uses the default or linear activation function (no activation function). The mean squared error (mse) loss is minimized when fitting the model.




Deep Learning CNN for Fashion-MNIST Clothing Classification

The Fashion-MNIST clothing classification problem is a new standard dataset used in computer vision and deep learning.


Dataset : https://github.com/zalandoresearch/fashion-mnist

You can directly import dataset from keras : from tf.keras.datasets import fashion_mnist

https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data




If you have any doubts refer to this URL 

https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

