import numpy as np
from scipy.sparse import csr_matrix

def similarity_space(X: csr_matrix, y: np.ndarray) -> csr_matrix:
    """
    Transforms a similarity matrix into a similarity space defined by class labels.

    This function calculates, for each sample, the sum of its similarities to all
    samples within each class.

    Args:
        X (csr_matrix): A symmetric (n_samples, n_samples) sparse matrix where X_ij
                        is the similarity between sample i and sample j.
                        Diagonal elements (X_kk) are assumed to be zero or not present.
        y (np.ndarray): A 1D array of shape (n_samples,) containing the class label
                        for each sample.

    Returns:
        csr_matrix: A sparse matrix Q of shape (n_samples, n_classes), where Q_rs is
                    the sum of similarities between sample r and all samples belonging
                    to class s.
    """
    # Ensure the input matrix is in CSR format for efficient row operations.
    if not isinstance(X, csr_matrix):
        X = X.tocsr()

    # Get the shape and find the unique classes and their integer indices.
    # `np.unique` with `return_inverse=True` is highly efficient for this.
    n_samples = X.shape[0]
    unique_labels, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(unique_labels)

    # Create the one-hot encoded class indicator matrix B.
    # B will be a sparse matrix of shape (n_samples, n_classes).
    # Each row `i` will have a single '1' in the column `j` corresponding
    # to the class of sample `i`.
    rows = np.arange(n_samples)
    cols = y_indices
    data = np.ones(n_samples, dtype=np.int8) # Use a memory-efficient dtype
    
    B = csr_matrix((data, (rows, cols)), shape=(n_samples, n_classes))

    # Perform the core operation using sparse matrix multiplication.
    # This is highly optimized and avoids any explicit Python loops.
    # The result Q_sparse will be a sparse matrix.
    Q_sparse = X @ B

    return Q_sparse