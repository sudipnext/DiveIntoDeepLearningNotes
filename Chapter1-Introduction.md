# 1. Introduction

## What is Machine Learning?
ML is a field of computer science that deals with the development of algorithms that are capable of learning from data and making predictions or decisions based on that data. It is a subset of artificial intelligence (AI) that focuses on the development of algorithms that can learn from and make predictions based on data.

## If machine learning is present why do we need deep learning?
Machine Learning is a broad field, for the patterns that aren't easy to capture or learn like images, videos, and audio, we need to use deep learning. Machine learning algorithms aren't able to learn these patterns effectively. If the problem is not deterministic, not follows a specific set of rules, and is not easy to capture, we need to use deep learning. 

## What is Deep Learning?
Deep learning is a subset of machine learning that uses neural networks to learn from data. It is a powerful tool for solving complex problems that are not easily solved by traditional machine learning algorithms.


## Exercises
### 1. Which part of code that you are currently writing could be "learned", i.e improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices? What data might you need to learn the designed behaviour?

> I am learning and using django these days and in django there are many design choices that can be learned, such as how to structure the models, views and the templates. There are also many heuristic design choices that can be learned, such as how to structure the URLs, how to handle the forms, and how to handle the authentication. The data that might be needed to learn the designed behaviour includes the models, views, templates, URLs, forms, and authentication. If we have enough data, we can learn the designed behaviour and improve the code automatically.

### 2. Which problems that you encounter have many examples of the desired behaviour, but no clear rules for how to achieve that behaviour? How could you use deep learning to solve these problems?

> The problem I specifically encountered which have many examples of the desired behaviour, but no clear rules is building recommendation systems. For example, in general the recommendation is a hard problem cause we don't know specifically what the user truly wants, but we can learn these from the user past behaviour and the data we have about the user. There is no fix pattern or rule that can be applied to all if we want to provide the recommendations that converts the user i.e the user clicks on the recommended item, buys it, or likes it. We can use deep learning to learn the patterns from the data we have about the user and provide recommendations that are more likely to convert the user.

### 3. Describe the relationships between algorithms, data, and computation. How do characteristics for the data and the current available computational resources influence the appropriateness of various algorithms?
> The relationships between the algorithms, data and computation is very much important and crucial. In summary as the algorithm complexity increases, the data size increases and the computational resources needed also increases and vice versa.

```
          Algorithms 
              / \
             /   \
            /     \
        Data _____ Computation
``` 

> The characteristics of the data and the current available computational resources influence the appropriateness of various algorithms in the following ways:

> **Data Characteristics**: The type of data (e.g., structured, unstructured, time-series, images) and its size can determine which algorithms are suitable. For example, deep learning algorithms are often used for unstructured data like images and text, while traditional machine learning algorithms may be more appropriate for structured data.

> **Computational Resources**: The available computational resources (e.g., CPU, GPU, memory) can limit the choice of algorithms. Some algorithms, like deep learning models, require significant computational power and memory, while others may be more efficient on limited resources. Eg: Graph Neural Networks (GNNs) are often used for graph data, but they can be computationally intensive and may require specialized hardware like GPUs for efficient training.

### 4. Name some settings where end-to-end training is not currently default approach but where it might be useful.
> End to End training in deep learning is a technique that uses directly the set of inputs and outputs to train the large neural networks. This is pretty useful in many settings, because it skips the intermediate steps and directly trains the model on the inputs and outputs. End to end learning requires large amount of training data than the traditional step by step approach for it to perform better in the real world scenario. The some settings where end-to-end training is not currently default approach but where it might be useful are:

> NLP

> Computer Vision

> Robotics

> Reinforcement Learning

> Autonomous Vehicles
