# Deep Learning IIITDMJ Notes
More on this here [Deep_learning_IIITDMJ.ipynb](https://github.com/sudipnext/DiveIntoDeepLearningNotes/blob/main/Deep_learning_IIITDMJ.ipynb)

# Day 1: Setting up the Environment and Introduction to Deep Learning

## Spectral Theories in Mathematics
> It involves breaking down the complex structure into smaller individual components, making them easier to understand.

```bash
[COMPLICATED] = [SIMPLE] + [SIMPLE] + [SIMPLE] + ...
```

### Deep Learning is Simple, Complicated, and Complex at the Same Time

> **Simple**: The basic concepts of deep learning are simple and easy to understand.  
> **Complicated**: Many parts: Deep learning involves many parts that are complicated and difficult to understand.  
> **Complex**: Due to the presence of non-linearities, deep learning is complex and difficult to understand.

## Terms and Objects in Math and Computers

### Linear Algebra Terminology

### Correspondence Between Math and Programming Terminology

| Domain   | Scalar     | Vector         | Matrix                     | Tensor                    |
|----------|------------|----------------|----------------------------|---------------------------|
| Math     | Scalar     | Vector         | Matrix                     | Tensor                    |
| PyTorch  | Float      | Tensor         | Tensor                     | Tensor                    |
| NumPy    | Array      | Array          | 2D Array                   | n-D Array                 |

> **Example**: A grayscale image is a 2D array, an RGB image is a 3D array, and a video is a 4D array (a sequence of images, where each image is a 2D array of pixels).

## Converting Reality into Data
> The process of converting reality into data is called **data collection**. It involves collecting data from various sources, such as sensors, cameras, and other devices, and converting it into a format that can be used for analysis and modeling.

### Two Types of Reality
> 1. **Continuous**  
> 2. **Discrete**

#### Continuous Reality
> 1. Numeric  
> 2. Real Numbers  
> 3. Many Values (Possibility of infinite values)  
> 4. Examples: Temperature, Height, Weight, etc.

#### Discrete Reality
> 1. Numeric  
> 2. Integer Numbers  
> 3. Few Values (Possibility of finite values)  
> 4. Examples: Number of students in a class, Number of cars in a parking lot, etc.

## Representing Categorical Data
> Categorical data can be represented using various techniques, including:  
> 1. **One-Hot Encoding**  
> 2. **Label Encoding**  
> 3. **Binary Encoding**

### One-Hot Encoding
> 1. 0 or 1 per category  
> 2. Each category is represented by a binary vector  
> 3. Example: If we have three categories: Red, Green, and Blue, then the one-hot encoding would be:  

```plaintext
Red: [1, 0, 0]
Green: [0, 1, 0]
Blue: [0, 0, 1]
```

| Genre | Action | Comedy | Drama |
|-------|--------|--------|-------|
| M1    | 1      | 0      | 0     |
| M2    | 0      | 1      | 0     |
| M3    | 0      | 0      | 1     |

> 4. Creates a sparse matrix.

### Label Encoding
> 1. Assigns a unique integer to each category.  
> 2. Example: If we have three categories: Red, Green, and Blue, then the label encoding would be:  

```plaintext
Red: 0
Green: 1
Blue: 2
```

### Binary Encoding
> 1. Converts the integer labels into binary format.  
> 2. Example: If we have three categories: Red, Green, and Blue, then the binary encoding would be:  

```plaintext
Red: 00
Green: 01
Blue: 10
```

## Vector and Matrix Transpose
> 1. **Transpose of a vector**: Converts a row vector into a column vector and vice versa.  
> 2. **Transpose of a matrix**: Flips the matrix over its diagonal, converting rows into columns and columns into rows.  

```plaintext
       T
[a b c]
```

Transpose:
```plaintext
[a]
[b]
[c]
```

## Dot Product
> The dot product is a mathematical operation that takes two equal-length sequences of numbers (usually coordinate vectors) and returns a single number. It is calculated by multiplying corresponding entries and summing those products.  

```plaintext
[a₁, a₂, a₃] ⋅ [b₁, b₂, b₃] = a₁b₁ + a₂b₂ + a₃b₃
```

## SoftMax Function
> The SoftMax function is a mathematical function that converts a vector of numbers into a probability distribution. It is often used in the output layer of a neural network for multi-class classification problems.  

$$
\text{SoftMax}(x) = \left[\frac{e^{z_i}}{\sum e^{z_i}} \text{ for } i \in \text{range(len(z))}\right]
$$

> SoftMax outputs values in the range [0, 1], where the sum of all outputs equals 1. This makes it suitable for multi-class classification problems, where each output represents the probability of a particular class.

## Logarithms
> Log is the inverse of the natural exponential function.  
> Log is a monotonic function, meaning it is always increasing.  
> This is important because minimizing \(x\) is the same as minimizing \(\log(x)\) (only for \(x > 0\)).  
> Log stretches small input values and compresses large input values.

![Logarithm Function Graph](images/logarithmic_function.png)

> The above graph shows the logarithmic function, which is a monotonic function that is always increasing. The logarithm of a number is the exponent to which the base must be raised to produce that number. For example, log base 10 of 100 is 2, because \(10^2 = 100\).

## Exponential Function
> The exponential function is a mathematical function that grows rapidly as the input increases. It is defined as:  

$$
\text{exp}(x) = e^x
$$

![Exponential Function Graph](images/exponential_function.png)

## Entropy
> Entropy is the measure of randomness or uncertainty in a system. In the context of information theory, it quantifies the amount of uncertainty in a random variable. The entropy of a discrete random variable \(X\) is defined as:  

$$
H(X) = -\sum p(x) \cdot \log(p(x)) \quad \text{for all } x \in X
$$

Where:  
- \(p(x)\) is the probability of \(x\) occurring.  
- \(x\) = data values  

> High entropy means lots of variability in the data, while low entropy means less variability in the data.  
> Low entropy means most of the values in the datasets repeat.  
> Entropy is nonlinear and makes no assumptions about the distribution of the data.

## Cross Entropy
> Cross-entropy is a measure of the difference between two probability distributions. It is often used in machine learning to quantify the difference between the predicted probability distribution and the true probability distribution. The cross-entropy between two probability distributions \(P\) and \(Q\) is defined as:  

$$
H(P, Q) = -\sum p(x) \cdot \log(q(x)) \quad \text{for all } x \in X
$$

Where:  
- \(p(x)\) is the true probability distribution,  
- \(q(x)\) is the predicted probability distribution.  

> It describes the relationship between the two probability distributions.

# Day 2: Mathematics for Deep Learning and Neural Network Basics

## Perceptron: A Linear Classifier
> A perceptron is a simple linear classifier that maps input features to output classes. It consists of a set of weights and a bias term, which are used to compute a weighted sum of the input features. The output is then passed through an activation function to produce the final prediction.

## Rosenblatt's Perceptron
> The perceptron was introduced by Frank Rosenblatt in 1958. It is a simple model that can be used to classify linearly separable data. The perceptron learns the weights and bias by minimizing the error between the predicted output and the true output.  

![alt text](image.png)  
Image source: [Rosenblatt's Perceptron](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2F%40aaronbrennan.brennan%2Fthe-perceptrons-beginning-rosenblatt-and-minsky-papert-1813def0817b&psig=AOvVaw0xmcABfuPQW-oZ-dJ-VuX4&ust=1749224390595000&source=images&cd=vfe&opi=89978449&ved=0CBgQjhxqFwoTCNil_eTO2o0DFQAAAAAdAAAAABAL)

> The output of a perceptron can be represented as:  

$$
\text{output} = \text{activation\_function}(w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n + b)
$$

Where:  
- \(w_1, w_2, \dots, w_n\) are the weights,  
- \(x_1, x_2, \dots, x_n\) are the input features,  
- \(b\) is the bias term,  
- \(\text{activation\_function}\) is a non-linear function that introduces non-linearity into the model.

## The XOR Problem
> The XOR problem is a classic example of a problem that cannot be solved by a single-layer perceptron. It is a non-linearly separable problem, meaning that it cannot be separated by a straight line in the input space. The XOR function takes two binary inputs and produces a binary output, where the output is true if the inputs are different and false if they are the same.  

```plaintext
XOR Truth Table:
| Input 1 | Input 2 | Output |
|----------|----------|--------|
|    0     |    0     |   0    |
|    0     |    1     |   1    |
|    1     |    0     |   1    |
|    1     |    1     |   0    |
```

> Rosenblatt's perceptron was criticized by Minsky and Papert in 1969 for not being able to solve the XOR problem. They showed that a single-layer perceptron cannot learn the XOR function, which led to a temporary decline in interest in neural networks.  

> Rosenblatt's perceptron proved that neural networks can learn by preparing an electrical model that can classify the digits 0-9 for a mail sorting machine.

### Perceptron Learning Algorithm | Criteria for Convergence
```plaintext
1. Initialize weights and bias to small random values.
2. For each training example (x, y):
   a. Compute the output: 
      output = activation_function(w1*x1 + w2*x2 + ... + wn*xn + b)
   b. Update weights and bias:
      w_i = w_i + learning_rate * (y - output) * x_i
      b = b + learning_rate * (y - output)
3. Repeat steps 2 until convergence (i.e., when the weights and bias do not change significantly).
      w_i = w_i + learning_rate * (y - output) * x_i
      b = b + learning_rate * (y - output)
3. Repeat steps 2 until convergence (i.e., when the weights and bias do not change significantly).
```

## Iterative Weight Update Rule by Learning

### Cauchy's Rule [1849]
> Cauchy's rule is an iterative method for finding the roots of a function. It is used to update the weights and bias in the perceptron learning algorithm. The rule states that the weights and bias should be updated in the direction of the negative gradient of the error function.  

```math
W^{(t+1)} = W^{(t)} - \eta \cdot \nabla E
```

```math
\text{Where:}  

W^{(t+1)} \text{ is the updated weight vector,}  
W^{(t)} \text{ is the current weight vector,}  
\eta \text{ is the learning rate (a small positive constant controlling the step size),}  
\nabla E \text{ is the gradient of the error function with respect to the weights.}  
```
#### Slope of the Error Function
> The slope of the error function is the gradient of the error function with respect to the weights. It indicates the direction in which the weights should be updated to minimize the error. The gradient is computed as:  

```math
\nabla E = \left[\frac{\partial E}{\partial w_1}, \frac{\partial E}{\partial w_2}, \ldots, \frac{\partial E}{\partial w_n}\right]
```
> Where:
- \( \frac{\partial E}{\partial w_i} \) is the partial derivative of the error function with respect to the weight \( w_i \).

> The gradient points in the direction of the steepest increase of the error function, so we update the weights in the opposite direction to minimize the error.

```math
W^{(t+1)} = W^{(t)} - \eta \cdot \nabla E
```

> Smaller \( \eta \) means slower convergence, while larger \( \eta \) means faster convergence but can lead to overshooting the minimum.

> -ve sign is because we want to minimize the error function, so we move in the direction of the negative gradient.

## Multi Layer Perceptron (MLP)
> A Multi-Layer Perceptron (MLP) is a type of neural network that consists of multiple layers of neurons, including an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to every neuron in the next layer, forming a fully connected network.

![MLP Diagram](image-1.png)

Image Reference[Datacamp](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning)

## Output of a Multi-Layer Perceptron
> The output of a Multi-Layer Perceptron is computed by passing the input through each layer, applying an activation function at each neuron, and finally producing the output at the output layer. The output can be represented as:
```math
\text{output} = \text{activation\_function}(W^{(L)} \cdot \text{activation\_function}(W^{(L-1)} \cdot \ldots \cdot \text{activation\_function}(W^{(1)} \cdot x + b^{(1)}) + b^{(2)}) + \ldots + b^{(L)})
```

Where:
- \( W^{(i)} \) is the weight matrix for layer \( i \),
- \( b^{(i)} \) is the bias vector for layer \( i \),
- \( \text{activation\_function} \) is a non-linear activation function applied at each layer,
- \( L \) is the number of layers in the MLP,
- \( x \) is the input vector.

## Activation Functions
> Activation functions introduce non-linearity into the model, allowing the MLP to learn complex relationships in the data. Common activation functions include:
### Sigmoid Function
> The sigmoid function is a smooth, S-shaped curve that maps any real-valued number to the range [0, 1]. It is defined as:
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```
> The sigmoid function is often used in the output layer of binary classification problems, as it produces a probability-like output.

### Hyperbolic Tangent Function (tanh)
> The hyperbolic tangent function is similar to the sigmoid function but maps the input to the range [-1, 1]. It is defined as:
```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```
> The tanh function is often used in hidden layers of MLPs, as it has a zero-centered output, which can help with convergence during training.

### Rectified Linear Unit (ReLU)
> The Rectified Linear Unit (ReLU) is a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero. It is defined as:
```math
\text{ReLU}(x) = \max(0, x)
```
> ReLU is widely used in hidden layers of MLPs due to its simplicity and effectiveness in mitigating the vanishing gradient problem.

### Leaky ReLU
> The Leaky ReLU is a variant of the ReLU that allows a small, non-zero gradient when the input is negative. It is defined as:
```math
\text{Leaky ReLU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
```
> Where \( \alpha \) is a small positive constant (e.g., 0.01). Leaky ReLU helps to prevent dead neurons during training.

### Softmax Function
> The Softmax function is used in the output layer of multi-class classification problems. It converts the raw output scores into probabilities that sum to 1. It is defined as:
```math
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, 2, \ldots, K
```
> Where \( K \) is the number of classes, and \( z_i \) is the raw output score for class \( i \). The Softmax function ensures that the output probabilities are non-negative and sum to 1.

### Swish Function
> The Swish function is a smooth, non-monotonic activation function that has been shown to perform well in deep neural networks. It is defined as:
```math
\text{Swish}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
```
> The Swish function is differentiable and has a non-zero gradient for all input values, which can help with convergence during training.
### Mish Function
> The Mish function is a smooth, non-monotonic activation function that has been shown to perform well in deep neural networks. It is defined as:
```math
\text{Mish}(x) = x \cdot \tanh(\text{Softplus}(x)) = x \cdot \tanh(\log(1 + e^x))
```
> The Mish function is differentiable and has a non-zero gradient for all input values, which can help with convergence during training.    

### Choice of Activation Function
> The choice of activation function depends on the specific problem and the architecture of the neural network.
> - For binary classification problems, the sigmoid function is often used in the output layer.
> - For multi-class classification problems, the Softmax function is commonly used in the output layer.
> - For hidden layers, ReLU and its variants (Leaky ReLU, Parametric ReLU, etc.) are widely used due to their simplicity and effectiveness in mitigating the vanishing gradient problem.

> Sigmoid can cause vanishing gradients, especially in deep networks, leading to slow convergence during training.
> Tanh is zero-centered, which can help with convergence, but it can still suffer from vanishing gradients in deep networks.
> ReLU is computationally efficient and helps mitigate the vanishing gradient problem, but it can suffer from dead neurons (neurons that never activate).

### Relating Sigmoid and Hyperbolic Tangent Functions: A Mathematical Derivation

The relationship between the sigmoid function and the hyperbolic tangent function is a fundamental concept in neural network theory. This derivation shows how the hyperbolic tangent can be expressed in terms of the sigmoid function.

Let's start with the observation that the hyperbolic tangent function can be derived from the sigmoid function through a simple transformation:

```math
2\sigma(a) - 1 = \tanh(a/2)
```

#### Mathematical Derivation

Beginning with the left side of the equation:

```math
2\sigma(a) - 1 = 2\left(\frac{1}{1 + e^{-a}}\right) - 1
```

```math
= \frac{2}{1 + e^{-a}} - 1
```

```math
= \frac{2}{1 + e^{-a}} - \frac{1 + e^{-a}}{1 + e^{-a}}
```

```math
= \frac{2 - 1 - e^{-a}}{1 + e^{-a}}
```

```math
= \frac{1 - e^{-a}}{1 + e^{-a}}
```

Now, to obtain the hyperbolic tangent form, we multiply both numerator and denominator by $e^{a/2}$:

```math
\frac{1 - e^{-a}}{1 + e^{-a}} = \frac{e^{a/2} - e^{-a/2}}{e^{a/2} + e^{-a/2}} = \tanh(a/2)
```

This elegant relationship shows that scaling and shifting the sigmoid function yields the hyperbolic tangent function. This connection explains why both functions are commonly used as activation functions in neural networks, with tanh providing a zero-centered output in the range [-1, 1], while sigmoid produces outputs in the range [0, 1].

## Solving the XOR Problem with Multi-Layer Perceptron

### XOR-1 Feature X'Formation | Kernel Trick
> The XOR problem can be solved using a Multi-Layer Perceptron (MLP) by transforming the input features into a higher-dimensional space. This transformation allows the MLP to learn non-linear decision boundaries that can separate the classes in the XOR problem.

| x2 | x1 | x3 = x1 * x2 | x1 xor x2 |
|----|----|--------------|-----------|
|  0 |  0 |      0       |     0     |
|  0 |  1 |      0       |     1     |
|  1 |  0 |      0       |     1     |
|  1 |  1 |      1       |     0     |

### XOR-2 from basic Logic Gates
