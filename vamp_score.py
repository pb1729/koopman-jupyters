import torch
import numpy as np


# DECORRELATION

def batched_2_moment(x, batch=64000):
    """ compute expected (outer product) square of tensor in batches
        x: (instances, dim)
        ans: (dim, dim) """
    instances, _ = x.shape
    ans = 0.
    for i in range(0, instances, batch):
        ans += torch.einsum("ix, iy -> xy", x[i:i+batch], x[i:i+batch])
    return ans/(instances - 1)

def batched_xy_moment(x, y, batch=64000):
    """ compute product moment of two tensors in batches
        x: (instances, dim1)
        y: (instances, dim2)
        ans: (dim1, dim2)"""
    instances, _ = x.shape
    assert y.shape[0] == instances
    ans = 0.
    for i in range(0, instances, batch):
        ans += torch.einsum("ix, iy -> xy", x[i:i+batch], y[i:i+batch])
    return ans/(instances - 1)

class Affine:
    """ Class representing an affine transformation in n-dimensional space """
    def __init__(self, W, mu):
        self.W = W
        self.mu = mu
    def __call__(self, x):
        return (self.W @ (x - self.mu).T).T
    def detach(self):
        return Affine(self.W.detach(), self.mu.detach())

def decorr(x, with_transform=True, epsilon=1e-6):
    """ decorrelate (i.e. apply a whitening transformation) to the data vector x with shape (batch, featuredim)
    dimensions where there is no variation will be discarded """
    batch, featuredim = x.shape
    mu = x.mean(0, keepdim=True)
    x = x - mu
    C = batched_2_moment(x)
    try:
        lam, Q = torch.linalg.eigh(C)
    except torch._C._LinAlgError as e:
        tag = np.random.randint(0, 1000)
        np.savetxt(f"error_dumps/{tag}_linalg_err.csv", C.detach().cpu().numpy(), delimiter=",")
        print(f"linalg error, random tag is {tag}")
        raise e
    keepdim_mask = lam > epsilon # discard dimensions where there is little variation (can't take -0.5 power of non-positive number)
    Q = Q[:, keepdim_mask]
    lam = lam[keepdim_mask]
    W = (1./torch.sqrt(lam)).reshape(-1, 1) * Q.T
    y = (W @ x.T).T
    if with_transform:
        return y, Affine(W, mu)
    return y


# VAMP SCORE
def vamp_score(chi_0, chi_1, mode="score", epsilon=1e-6):
    """ function computing the VAMP-2 score, implemented in torch for differentiability
    chi_0 - right feature vector, shape is (batch, featuredim)
    chi_1 - left feature vector, shape is (batch, featuredim)
    mode - one of several options:
        * "score": return the VAMP score directly (i.e. the Frobenius norm of the matrix)
        * "matrix": just return the estimated truncated Koopman operator (in f,g space) C_10
        * "all": return the decorrelation transformations, and also C_10
    epsilon - eigenvalues of covariance matrix are clipped to be at least this, for numberical stability """
    batch, featuredim = chi_0.shape
    assert chi_1.shape == chi_0.shape
    f, trans_0 = decorr(chi_0) # f.shape: (batch, truncdim_f <= featuredim)
    g, trans_1 = decorr(chi_1) # g.shape: (batch, truncdim_g <= featuredim)
    del chi_0, chi_1
    # compute covariance (since f, g should now have mean 0)
    C_10 = batched_xy_moment(g, f) # estimated truncated Koopman operator
    if mode == "score":
        return (C_10**2).sum()
    elif mode == "matrix":
        return C_10
    elif mode == "all":
        return trans_0, trans_1, C_10
    else:
        assert False, "invalid mode passed"



if __name__ == "__main__":
    # test batched eval
    x = torch.randn(20, 13)
    c1 = torch.cov(x)
    c2 = batched_cov(x.T, x.T, 10)
    print(abs(c1 - c2).max())


