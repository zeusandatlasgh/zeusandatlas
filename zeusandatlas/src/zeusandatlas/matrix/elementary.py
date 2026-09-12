import torch

def rowswap(matrix, source_row, target_row):
    M = matrix.clone()
    M[[source_row, target_row]] = M[[target_row, source_row]]
    return M

def rowscale(matrix, row, scale):
    M = matrix.clone()
    M[row] = M[row] * scale
    return M

def rowreplacement(matrix, row_i, row_j, j, k):
    M = matrix.clone()
    scaled_i = rowscale(M, row_i, j)[row_i]
    scaled_j = rowscale(M, row_j, k)[row_j]
    M[row_i] = scaled_i + scaled_j
    return M

def rref(matrix):
    M = matrix.clone().to(torch.float64)
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        nonzero_row = None
        for r in range(pivot_row, rows):
            if abs(M[r, col].item()) > 1e-10:
                nonzero_row = r
                break
        if nonzero_row is None:
            continue
        if nonzero_row != pivot_row:
            M = rowswap(M, pivot_row, nonzero_row)
        pivot_val = M[pivot_row, col].item()
        M = rowscale(M, pivot_row, 1.0 / pivot_val)
        for r in range(rows):
            if r != pivot_row:
                factor = M[r, col].item()
                if abs(factor) > 1e-10:
                    M = rowreplacement(M, r, pivot_row, 1.0, -factor)
        pivot_row += 1
    return M


if __name__ == "__main__":
    A = torch.tensor([
        [1.0, 3.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, -4.0]
    ])
    step1 = rowswap(A, 0, 1)
    print("After R1 <-> R2:\n", step1)
    step2 = rowscale(step1, 0, 1/3)
    print("After (1/3)R1:\n", step2)
    step3 = rowreplacement(step2, 2, 0, 1.0, -3.0)
    print("After R3 = -3R1 + R3:\n", step3)
    print("Full RREF:\n", rref(A))
