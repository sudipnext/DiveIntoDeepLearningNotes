## 2.1 Data Manipulation
In order to get anything done in machine learning, deep learning or anything we need to learn the art of data manipulation. This includes reading data from files, manipulating data in memory, and writing data back to files. The most common libraries for data manipulation in Python are NumPy and Pandas.

Pytorch is a python library for deep learning that is built on top of NumPy. It provides a high-level interface for building and training neural networks, as well as a low-level interface for manipulating tensors (multi-dimensional arrays).

## 2.1.1 Pytorch
importing pytorch in the code is done as follows:
```python
import torch
```
### Tensors
Tensors are the fundamental data structure in PyTorch. They look similar to NumPy arrays, but they are used inside the pytorch library. Tensors can be created from NumPy arrays, Python lists, or directly using PyTorch functions. Tensors are multidimensional arrays that can be used to represent data in a variety of formats, including images, text, and audio.

### Creating Tensors
```python
import torch
# Creating a tensor from a list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
# Creating a tensor from a NumPy array
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.from_numpy(numpy_array)
# Creating a tensor with random values
```
Let's see some examples of creating tensors in PyTorch:

We can use `arange` to create a tensor with a range of values:
```python
tensor_range = torch.arange(0, 10, 1)  # Creates a tensor with values from 0 to 9 and the last one is the step size
#Result:
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```
Counting the no. of elements in a tensor can be done using the `numel()` method:
```python
num_elements = tensor_range.numel()  # Returns the number of elements in the tensor
#Result:
10
```

We can access the shape of the tensor by calling the `shape` attribute:
```python
tensor_shape = tensor_range.shape  # Returns the shape of the tensor
# Result:
torch.Size([10])  # This indicates that the tensor has 10 elements in a single dimension
```

We can change the shape of the tensor by using the `reshape` method:
```python
reshaped_tensor = tensor_range.reshape(2, 5)  # Reshapes the tensor to a 2x5 matrix i.e 2 rows and 5 columns
# Result:
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
```

If we want our pytorch to take care of one of the dimensions then we can use `-1` in the reshape method:
```python
reshaped_tensor_auto = tensor_range.reshape(2, -1)  # Automatically infers the second dimension
# Result:
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
```
Or 
```python
reshaped_tensor_auto = tensor_range.reshape(-1, 5)  # Automatically infers the first dimension
# Result:
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
```
More on this here [Chapter2.ipynb](https://github.com/sudipnext/DiveIntoDeepLearningNotes/blob/main/Chapter2.ipynb)


## 2.2 Data Preprocessing



## 2.3 Linear Algebra


## 2.4 Calculus


## 2.5 Automatic Differentiation


## 2.6 Probability and Statistics


## 2.7 Documentation