
## Day2 Python For Deep Learning

### min, max, argmin, argmax
Let's understand the concepts of `min`, `max`, `argmin`, and `argmax` using Python.

```python
import numpy as np
# Create a sample array
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
# Find the minimum value
min_value = np.min(arr)
# Find the maximum value
max_value = np.max(arr)
# Find the index of the minimum value
argmin_index = np.argmin(arr)
# Find the index of the maximum value
argmax_index = np.argmax(arr)
# Print the results
print("Minimum value:", min_value)
print("Maximum value:", max_value)
print("Index of minimum value:", argmin_index)
print("Index of maximum value:", argmax_index)

```
#### Output:
```plaintext
Minimum value: 1
Maximum value: 9
Index of minimum value: 1
Index of maximum value: 5

```
Formulae of Argmin and Argmax

```math
z = \underset{x}{\operatorname{arg}}  min f(x)
```

```math
w = \underset{x}{\operatorname{arg}}  max f(x)
```
### Application of Argmin and Argmax in Deep Learning
> Let's say we have a task in NN to classify the image of a sign board into one of the classes: `STOP`, `GO`, `YIELD`. The output layer of the neural network will have three neurons, each representing one of these classes. The output of the network will be a vector of probabilities for each class.

```python
import numpy as np

# Sample output probabilities from the neural network
output_probs = np.array([0.7, 0.2, 0.1])

# Find the predicted class using argmax
predicted_class = np.argmax(output_probs)

# Print the predicted class
print("Predicted class:", predicted_class)
```
#### Output:
```plaintext
Predicted class: 0
```
> In this example, the `argmax` function is used to find the index of the maximum value in the output probabilities, which corresponds to the predicted class. In this case, the predicted class is `0`, which could represent the `STOP` sign.

## Mean and Variance

### Mean
> The mean is the average of a set of values. It is calculated by summing all the values and dividing by the number of values. The mean is a measure of central tendency and provides an indication of the typical value in a dataset.

```python
import numpy as np
# Sample data
data = np.array([1, 2, 3, 4, 5])

# Calculate the mean
mean = np.mean(data)

# Print the mean
print("Mean:", mean)
```
#### Output:
```plaintext
Mean: 3.0
```
> Mean gives the central tendency of the data, which is useful for understanding the overall distribution of values.

Formulae
```math
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
```
Where,
μ is the mean,
N is the number of values,
xi is the i-th value in the dataset.

### Concept of Dispersion
> Dispersion refers to the spread or variability of a set of values. It provides insights into how much the values deviate from the mean. Common measures of dispersion include variance and standard deviation.
### Variance
> Variance is a measure of how much the values in a dataset deviate from the mean. It is calculated by taking the average of the squared differences between each value and the mean. Variance provides an indication of the spread of the data.

Formulae
```math 
\sigma^2 = \frac{1}{1-N} \sum_{i=1}^{N} (x_i - \mu)^2
```
Where:
- σ is the variance,
- N is the number of values,
- x_i is the i-th value in the dataset,
- μ is the mean of the dataset.
