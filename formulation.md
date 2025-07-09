### Overview

The operation transforms a given similarity matrix into a "similarity space" based on class labels. This process aggregates similarity scores according to class membership. Specifically, given a matrix where each entry represents the similarity of a sample to a reference item, it computes a new matrix where each entry represents the total similarity of that sample to an entire class of reference items.

### Mathematical Formulation

Let there be a set of $n$ evaluated samples and a set of $m$ reference samples. The inputs are:

1.  An $n \times m$ **similarity matrix**, denoted as $K$. An element $K_{ij}$ represents a pre-computed measure of similarity between the $i$-th evaluated sample and the $j$-th reference sample.
2.  A **label vector**, denoted as $\mathbf{y} = (y_1, y_2, \dots, y_m)$, where $y_j$ is the class label assigned to the $j$-th reference sample.
3.  A **set of unique and sorted classes**, $\mathcal{C} = \{c_1, c_2, \dots, c_p\}$, where $p$ is the total number of distinct classes.

The goal is to compute an $n \times p$ matrix, which we will call $Q$, that represents the data in the new similarity space.

#### 1. Defining the Similarity Space

An element $Q_{ik}$ in the target matrix $Q$ is defined as the sum of similarities from the $i$-th evaluated sample to all reference samples that belong to class $c_k$. This can be expressed using the following summation:

$$Q_{ik} = \sum_{j=1}^{m} K_{ij} \cdot \mathbb{I}(y_j = c_k)$$

where $\mathbb{I}(\cdot)$ is the **indicator function**:

$$
\mathbb{I}(\text{condition}) =
\begin{cases}
1 & \text{if condition is true} \\
0 & \text{if condition is false}
\end{cases}
$$

This formula effectively filters the similarity values in each row of $K$, summing only those that correspond to a specific class.

#### 2. Formulation as Matrix Multiplication

The summation above can be expressed more elegantly and computed more efficiently as a matrix product. To achieve this, we first construct a **class indicator matrix**, which we will denote as $Y$.

This matrix $Y$ has dimensions $m \times p$. Each element $Y_{jk}$ is 1 if the $j$-th reference sample belongs to class $c_k$, and 0 otherwise.

$$Y_{jk} = \mathbb{I}(y_j = c_k)$$

The matrix $Y$ acts as a mapping from the reference samples to the classes.

The final similarity space matrix $Q$ is then simply the product of the original similarity matrix $K$ and the class indicator matrix $Y$:

$$Q = K Y$$

To verify this, consider the element $(KY)_{ik}$ of the resulting product matrix:

$$(KY)_{ik} = \sum_{j=1}^{m} K_{ij} Y_{jk} = \sum_{j=1}^{m} K_{ij} \cdot \mathbb{I}(y_j = c_k)$$

This confirms that the matrix product yields the desired similarity space matrix $Q$. This formulation transforms the problem from explicit summation into a standard linear algebra operation.