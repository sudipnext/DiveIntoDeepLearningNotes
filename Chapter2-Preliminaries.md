Jupyter Notebook: [Chapter2.ipynb](https://github.com/sudipnext/DiveIntoDeepLearningNotes/blob/main/Chapter2.ipynb)


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


## 2.2 Data Preprocessing
> Data Preprocessing is the process of transforming raw data into a format that is suitable for analysis. This includes cleaning, transforming, and normalizing the data.
### 2.2.1 Reading the Dataset
We can read datasets from various sources such as CSV files, Excel files, SQL databases, and more. The most common library for reading datasets in Python is Pandas.

Let's imagine we have dataset like this table:
| Feature1 | Feature2 | Feature3 | Target |
|----------|----------|----------|--------|
| 1.0      | 2.0      | 3.0      | 0      |
| 4.0      | 5.0      | 6.0      | 1      |
We can read this dataset using Pandas as follows:

```python
import pandas as pd
# Reading a CSV file
df = pd.read_csv('data.csv')
# Reading an Excel file
df = pd.read_excel('data.xlsx')
```
### 2.2.2 Data Preparation
Data preparation is the process of cleaning and transforming the data to make it suitable for analysis. This includes handling missing values, removing duplicates, and transforming categorical variables into numerical variables.

```python
inputs, targets = df.iloc[:, :-1], df.iloc[:, -1]  # Splitting the dataset into inputs and targets
# removing missing values
inputs = inputs.dropna()  # Dropping rows with missing values in inputs
targets = targets.dropna()  # Dropping rows with missing values in targets
# removing duplicates
inputs = inputs.drop_duplicates()  # Dropping duplicate rows in inputs
targets = targets.drop_duplicates()  # Dropping duplicate rows in targets
# transforming categorical variables into numerical variables
inputs = pd.get_dummies(inputs)  # One-hot encoding categorical variables in inputs
```
### 2.2.3 Conversion to the Tensor Format
After preparing the data, we need to convert it into a format that can be used by PyTorch. This involves converting the Pandas DataFrame into a PyTorch tensor.

```python
import torch
# Converting the inputs and targets to PyTorch tensors
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32)  # Converting inputs to a tensor
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)  # Converting targets to a tensor
```
### Exercises
1. Loading the UCI ML repository Abalone dataset and inspecting the properties of the dataset. What fraction of the data has missing values? What fraction of the variables are numerical, categorical, or text?
> Only one variable is categorical, which is the `Sex` variable. The rest are numerical variables. The dataset has no missing values.

## 2.3 Linear Algebra


## 2.4 Calculus


## 2.5 Automatic Differentiation


## 2.6 Probability and Statistics


## 2.7 Documentation