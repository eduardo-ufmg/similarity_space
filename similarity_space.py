from typing import cast

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def similarity_space(
    K: csr_matrix, y: np.ndarray, classes: np.ndarray | None = None
) -> np.ndarray:
    """
    Calculates the similarity space matrix from a kernel matrix and labels.

    This function transforms a kernel matrix `K` into a similarity space `Q`.
    The transformation is performed by aggregating the kernel values for each
    sample based on the class labels of the reference samples. This approach is
    highly efficient, leveraging sparse matrix multiplication.

    Parameters:
        K (csr_matrix): A sparse matrix of shape (n_samples, n_references).
                        The element X_ij is the kernel value from reference
                        sample `x_j` to evaluated sample `x_i`.
        y (np.ndarray): An array of shape (n_references,) containing the labels
                        for the reference samples `x_j`.
        classes (np.ndarray, optional): A sorted array of unique class labels
                        to use for the output space. If provided, the output
                        matrix `Q` will have a column for each class in this
                        array. If None (default), the classes are inferred
                        from the unique labels present in `y`.

    Returns:
        np.ndarray: A dense matrix `Q` of shape (n_samples, n_classes).
                    An element Q_ic represents the sum of kernel values from all
                    reference samples belonging to class `c` to the sample `x_i`.
    """
    # --- Input Validation ---
    if not isinstance(K, csr_matrix):
        raise TypeError(f"Input K must be a scipy.sparse.csr_matrix, but got {type(K)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input y must be a numpy.ndarray, but got {type(y)}")

    n_samples, n_references = K.shape  # type: ignore

    if n_references != len(y):
        raise ValueError(
            f"Shape mismatch: K.shape[1] ({n_references}) must equal len(y) ({len(y)})"
        )

    # --- Step 1: Map labels to integer indices ---
    if classes is None:
        # Infer classes from the provided labels `y`.
        # `classes` will be sorted, and `class_indices` will map each `y`
        # element to its index in the new `classes` array.
        classes, class_indices = np.unique(y, return_inverse=True)
    else:
        # Use the provided `classes` array.
        # This is useful when `y` might not contain all possible classes.
        if not isinstance(classes, np.ndarray):
            raise TypeError(
                f"Input 'classes' must be a numpy.ndarray, but got {type(classes)}"
            )
        if not np.all(classes[:-1] < classes[1:]):
            raise ValueError("The provided 'classes' array must be sorted and unique.")

        # Check that all labels in `y` are actually present in `classes`.
        unique_y_labels = np.unique(y)
        if not np.all(np.isin(unique_y_labels, classes)):
            missing = unique_y_labels[~np.isin(unique_y_labels, classes)]
            raise ValueError(
                f"Labels {list(missing)} from y are not in the provided classes array."
            )

        # Map each label in `y` to its index in the `classes` array.
        # `np.searchsorted` is highly efficient for this mapping.
        class_indices = np.searchsorted(classes, y)

    n_classes = len(classes)

    # --- Step 2: Create a sparse "indicator" matrix ---
    # The core of the efficient solution is to create a matrix, Y_indicator,
    # of shape (n_references, n_classes) that "selects" the right columns.
    # An entry (j, c) in this matrix is 1 if reference sample j belongs to
    # class c, and 0 otherwise.
    row_indices = np.arange(n_references)
    col_indices = class_indices
    data = np.ones(n_references, dtype=K.dtype)

    # We build the matrix in Compressed Sparse Column (CSC) format because
    # multiplication of a CSR matrix (K) by a CSC matrix is highly optimized.
    Y_indicator = csc_matrix(
        (data, (row_indices, col_indices)), shape=(n_references, n_classes)
    )

    # --- Step 3: Perform matrix multiplication ---
    # The desired output Q is the product of K and Y_indicator.
    # The element (i, c) of Q is the sum of similarities from sample `i`
    # to all reference samples belonging to class `c`.
    Q_sparse = K @ Y_indicator

    # --- Step 4: Convert result to a dense array ---
    # The operation results in a sparse matrix. We convert it to a dense
    # numpy array as specified by the function's return type.
    return cast(csc_matrix, Q_sparse).toarray()
