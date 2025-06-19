import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def similarity_space(X: csr_matrix, y: np.ndarray) -> np.ndarray:
    """
    Calculates the similarity space matrix from a kernel matrix and labels.

    This function transforms a kernel matrix `X` into a similarity space `Q`.
    The transformation is performed by aggregating the kernel values for each
    sample based on the class labels of the reference samples. This approach is
    highly efficient, leveraging sparse matrix multiplication to avoid Python loops
    and minimize memory overhead.

    Args:
        X (csr_matrix): A sparse matrix of shape (n_samples, n_references).
                        The element X_ij is the value of an RBF kernel evaluated
                        from the reference sample `x_j` to the evaluated sample `x_i`.
        y (np.ndarray): An array of shape (n_references,) containing the labels
                        for the reference samples `x_j`.

    Returns:
        np.ndarray: A dense matrix `Q` of shape (n_samples, n_classes).
                    An element Q_ic represents the sum of kernel values from all
                    reference samples belonging to class `c` to the sample `x_i`.
    """
    # --- Input Validation ---
    if not isinstance(X, csr_matrix):
        raise TypeError(f"Input X must be a scipy.sparse.csr_matrix, but got {type(X)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input y must be a numpy.ndarray, but got {type(y)}")
    
    n_samples, n_references = X.shape #type: ignore
    
    if n_references != len(y):
        raise ValueError(
            f"Shape mismatch: X.shape[1] ({n_references}) must equal len(y) ({len(y)})"
        )

    # --- Step 1: Map labels to integer indices ---
    # `np.unique` with `return_inverse=True` is perfect for this.
    # `classes` will store the unique labels in sorted order.
    # `class_indices` will be an array where each element is the integer index
    # corresponding to the label in the original `y` array.
    classes, class_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    # --- Step 2: Create a sparse "indicator" matrix ---
    # The core of the efficient solution is to create a matrix, Y_indicator,
    # of shape (n_references, n_classes) that "selects" the right columns of X
    # for summation. An entry (j, c) in this matrix is 1 if reference sample j
    # belongs to class c, and 0 otherwise. This is essentially a sparse one-hot
    # encoding of the labels in `y`.
    
    # We provide the data and the (row, column) coordinates for the non-zero elements.
    row_indices = np.arange(n_references)
    col_indices = class_indices
    data = np.ones(n_references, dtype=X.dtype)  # Use X's dtype for compatibility

    # We build the matrix in Compressed Sparse Column (CSC) format because
    # multiplication of a CSR matrix by a CSC matrix is a highly optimized operation.
    Y_indicator = csc_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_references, n_classes)
    )

    # --- Step 3: Perform matrix multiplication ---
    # The desired output Q is simply the product of X and Y_indicator.
    # The element (i, c) of the resulting matrix Q is calculated as:
    # Q_ic = sum_{j=0 to n_references-1} (X_ij * Y_indicator_jc)
    # Since Y_indicator_jc is 1 only when y_j belongs to class c, this simplifies to:
    # Q_ic = sum(X_ij) for all j where y_j == c
    # This is precisely the required computation.
    Q_sparse = X @ Y_indicator

    # --- Step 4: Convert result to a dense array ---
    # The operation results in a sparse matrix. We convert it to a dense
    # numpy array as specified by the function's return type.
    return Q_sparse.toarray()
