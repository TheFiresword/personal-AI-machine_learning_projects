# Machine learning models from scratch
![Machine Learning](https://github.com/me50/TheFiresword/blob/insa/2023-06/mini-ml-library/Image1.png?raw=true) ![Machine Learning](https://github.com/me50/TheFiresword/blob/insa/2023-06/mini-ml-library/Image2.png?raw=true)

## From A to Z
This mini machine learning library has been designed entirely from scratch. Each algorithm, including backpropagation for deep learning, has been custom implemented to guarantee total control over the learning and prediction process.

## Cutting-edge algorithms
This library incorporates some of the most advanced algorithms in machine learning, including backpropagation for deep neural networks. I've designed these algorithms using the latest research in data science to guarantee outstanding performance.

## Features
- Initialize machine learning models
- Train models on customized datasets
- Evaluate model performance with common metrics
- Support for classification, linear regression and logistic regression

## Model training example
Here is an example of how you can train a logistic regression model with our library, using backpropagation to optimize the model weights:

```python
from my_ai_utils import *

df = pd.read_csv("iris.data.csv", header=None, names=['sepal_length', 'sepal_width',
                                          'petal_length', 'petal_width', 'category'])
df.describe()
#===========================================
# Dataset visualization
#===========================================
real_set = np.array(df)
size = len(real_set)
for e in range(0, 2):
    for j in range(1, 4):
        if e != j:
            category1 = np.array([(real_set[:, e][i], real_set[:, j][i]) for i in range(size) if real_set[:, -1][i] == "Iris-setosa"])
            category2 = np.array([(real_set[:, e][i], real_set[:, j][i]) for i in range(size) if real_set[:, -1][i] == "Iris-versicolor"])
            category3 = np.array([(real_set[:, e][i], real_set[:, j][i]) for i in range(size) if real_set[:, -1][i] == "Iris-virginica"])

            f = plt.figure()
            plt.plot(category1[:, 0], category1[:, 1], '+')
            plt.plot(category2[:, 0], category2[:, 1], 'o')
            plt.plot(category3[:, 0], category3[:, 1], '*')

            f.show()

#===========================================
# Create then Train then Test the model
#===========================================
real_set = np.array(pd.read_csv("iris.data.csv"))
np.random.shuffle(real_set)

size = len(real_set)
X_set = real_set[:,:-1]
Y_set = real_set[:,-1]

train_and_validation_size = int(size*(70+15)/100)
_train_and_validation_set = X_set[:train_and_validation_size, :]
_train_and_validation_y = Y_set[:train_and_validation_size]

_test_set = X_set[train_and_validation_size: , :]
_test_y = Y_set[train_and_validation_size:]

#===========================================
#  Training
#===========================================
iris_model = NetModel(input_shape=(4, ), usage="MultiClassification")
iris_model.add_layer(Dense(4, activation_function="sigmoid"))
iris_model.compile(3, categories=['Iris-setosa','Iris-versicolor', 'Iris-virginica'], output_function="softmax", initializer="xavier")
iris_model.train(_train_and_validation_set, _train_and_validation_y,"quadratic", nepochs= 500, learning_rate=0.01, validation_per=15/(70+15))
iris_model.display_losses()
z = iris_model.predict_sample(_test_set,_test_y, metric="accuracy")

