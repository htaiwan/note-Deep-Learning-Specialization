# Logistic Regression with a Neural Network mindset

## 1. Packages
首先，要先知道在這project中用了哪些package。

- [numpy](www.numpy.org): 用來處理向量的運算。
- [h5py](http://www.h5py.org): 用來處理讀取H5的檔案資料，用來讀取測試資料
- [matplotlib](http://matplotlib.org): 用來顯示圖表。
- [PIL](http://www.pythonware.com/products/pil/), [scipy](https://www.scipy.org/): 用來測試model，跟自己輸入的圖片。

```python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
```

## 2. Overview of the Problem set
永遠要先了解要處理的資料格式，並做一定程度資料整理(preprocessing)。

- training set 有 m_train 個圖片並標示是否為貓(y=1)或不是貓(y=0)
- test set 有 m_test 個圖片並標示是否為貓(y=1)或不是貓(y=0)
- 每張圖片的shape (num_px, num_px, 3) 其中 3 標示是 (RGB). 因此每張圖片是正方形(height = num_px)且(width = num_px)

```python
# 讀取data，load_dataset()是利用h5py來讀取H5的資料
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 了解每個data set的shape
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Number of training examples: m_train = 209
# Number of testing examples: m_test = 50
# Height/Width of each image: num_px = 64
# Each image is of size: (64, 64, 3)
# train_set_x shape: (209, 64, 64, 3)
# train_set_y shape: (1, 209)
# test_set_x shape: (50, 64, 64, 3)
# test_set_y shape: (1, 50)
```

- 進行資料處理
 - **Reshape**: the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px  ∗∗  num_px  ∗∗ 3, 1).
 - ```python
X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
```

```python
# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# train_set_x_flatten shape: (12288, 209)
# train_set_y shape: (1, 209)
# test_set_x_flatten shape: (12288, 50)
# test_set_y shape: (1, 50)
```
 - **center and standardize your dataset**: for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255

```python
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

<font color='red'>

**重點:**

Datasets的先處理(pre-processing):

- 了解Datasets的 dimensions 和 shapes。
- 進行Datasets的reshape。
- 進行Datasets的standardize。

</font color='red'>

## 3. General Architecture of the learning algorithm
- Logistic Regression其實就一個簡單的Neural Network。

![8](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/8.png)

- **Mathematical expression of the algorithm**:

![9](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/9.png)

<font color='red'>

**重點:**
 
- 初始化model中的參數。
- 進行參數學習，根據新的參數來不斷降低cost function。  
- 利用學習到的參數來進行資料預測。
- 分析結果，並作結論。

</font color='red'>


## 4. Building the parts of our algorithm

- 建立一個Neural Network的主要步驟如下:
	- 1. 定義model structure (例如 number of input features) 
	- 2. model's parameters的初始化
	- 3. 迴圈:
   		- 計算目前loss (forward propagation)
    	- 計算目前gradient (backward propagation)
    	- 更新arameters (gradient descent)

通常 1-3 會各自獨立寫成一個function，最後再整合到`model()`中。

### 4.1 Helper functions

``` python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1 / (1+np.exp(-z))
    
    return s
```

### 4.2 Initializing parameters

``` python
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w, b = np.zeros((dim, 1)), 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
```

### 4.3 Forward and Backward propagation

![10](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/10.png)

``` python
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)  # compute activation                              
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) # compute cost                               
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```

### 4.4. Optimization

- 已經將parameters初始化。
- 也計算出cost function 和 gradient。
- 現在要利用gradient descent來更新這些參數( θ=θ−α dθ )。

``` python
# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```

- 已經利用上面的function來學到參數 w 和 b，接下來該要進行預測。

![11](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/11.png)

``` python
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T,X)+b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if(A[0,i] <= 0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
```

<font color='red'>

**重點:**
 
- 參數w和b的初始化。
- 利用迴圈不斷來訓練參數。
	- 計算cost和gradient。
	- 利用gradient descent進行參數更新。	   
- 利用學習到的參數w和b來進行資料預測。

</font color='red'>

## 5. Merge all functions into a model

```python
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
        
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
```

```python
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
```
![12](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/12.png)

- 進行cost function 和 gradients的繪製。

```python
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
```
![13](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/13.png)

## 6. Further analysis
### 6.1 Choice of learning rate

- 選擇一個正確的learning rate是很重要的。
	- learning rate太大，則參數可能不會學到最佳值。
	- learning rate太小，則學習速度會很慢。

```python
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

![14](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/14.png)

![15](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/15.png)

<font color='red'>

**本作業重點:**
 
- dataset的preprocssing是很重要的。
- 將每個function分開獨立，最後才整合在model中。   
- learning rate的設定，對演算法的效率影響很大。

</font color='red'>