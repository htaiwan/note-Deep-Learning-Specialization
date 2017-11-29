#  1. Logistic Regression as a Neural Network
## Binary Classification
> * In a binary classification problem, the result is a discrete value output.
> 
> * Example: Cat vs Non-Cat
> > * The goal is to train a classifier that the input is an image represented by a feature vector, 𝑥, and predicts whether the corresponding label 𝑦 is 1 or 0. In this case, whether this is a cat image (1) or a non-cat image (0).

![8](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/8.png)

> > * To create a feature vector, 𝑥, the pixel intensity values will be “unroll” or “reshape” for each color. The dimension of the input feature vector 𝑥 is 𝑛𝑥 = 64 𝑥 64 𝑥 3 = 12 288

![9](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/9.png)

## Logistic Regression
> * **Logistic regression** is a learning algorithm used in a supervised learning problem when the output 𝑦 are all either zero or one. 
> * The goal of logistic regression is to minimize the error between its predictions and training data.

![10](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/10.png)

## Logistic Regression Cost Function
> * **Hypothesis function:**

![11](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/11.png)

> * **Loss (error) function:** 
> 
> > * The loss function measures the discrepancy between the prediction (𝑦̂(𝑖)) and the desired output (𝑦(𝑖)). In other words, the loss function computes the error for a single training example.

![12](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/12.png)

> * **Cost function:**
> 
> > * The cost function is the average of the loss function of the entire training set. 
> > * We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function. (<-- Gradient Descent)

![13](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/13.png)

## Gradient Descent

![14](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/14.png)

## Derivatives

![15](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/15.png)

## Computation graph

![16](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/16.png)

## Derivatives with a Computation Graph

![17](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/17.png)

## Logistic Regression Gradient Descent

![18](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/18.png)

## Gradient Descent on m Examples

![19](https://github.com/htaiwan/note-Deep-Learning-Specialization/blob/master/Assets/19.png)

## Explanation of logistic regression cost function (optional)


# 2. Python and Vectorization

## Vectorization
> * 透過GPU或CPU執行平行化運算，速度會遠比執行loop快很多。
> * 

## Vectorizing Logistic Regression

## Vectorizing Logistic Regression's Gradient Output

## Broadcasting in Python

## A note on python/numpy vectors

