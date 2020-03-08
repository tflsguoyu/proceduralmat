import torch as th
import numpy as np

def lookup1d(A, P):
    # assert(A.dim() > 1 and P.dim() == 1)
    n = A.size()[0]

    P_int = P.floor()
    W = P - P_int
    P_int = P_int.type(th.int64)

    B1 = A[P_int]
    w1 = 1-W

    B2 = A[P_int+1]
    w2 = W

    B = B1 * w1.unsqueeze(-1).expand_as(B1) \
      + B2 * w2.unsqueeze(-1).expand_as(B2)

    return B

def lookup2d(A, P):
    assert(A.dim() > 1)
    n = A.size()[0]

    P_int = P.floor()
    W = (P - P_int).clamp(0,1)
    P_int = P_int.type(th.long)

    Pi, Pj = P_int[0], P_int[1]
    wi0, wj0 =   W[0],   W[1]
    wi1, wj1 = 1-W[0], 1-W[1]

    B1 = A[Pi%n, Pj%n]
    w1 = wi1 * wj1

    B2 = A[Pi%n, (Pj+1)%n]
    w2 = wi1 * wj0

    B3 = A[(Pi+1)%n, Pj%n]
    w3 = wi0 * wj1

    B4 = A[(Pi+1)%n, (Pj+1)%n]
    w4 = wi0 * wj0

    B = B1 * w1.unsqueeze(2).expand_as(B1) \
      + B2 * w2.unsqueeze(2).expand_as(B2) \
      + B3 * w3.unsqueeze(2).expand_as(B3) \
      + B4 * w4.unsqueeze(2).expand_as(B4)

    return B

def lookup3d(A, P):
    assert(A.dim() > 1)
    n = A.size()[0]

    P_int = P.floor()
    W = P - P_int
    P_int = P_int.type(th.int64)

    Pi, Pj, Pk = P_int[0], P_int[1], P_int[2]
    wi0, wj0, wk0 =   W[0],   W[1],   W[2]
    wi1, wj1, wk1 = 1-W[0], 1-W[1], 1-W[2]

    B1 = A[Pi, Pj, Pk]
    w1 = wi1 * wj1 * wk1

    B2 = A[Pi, Pj, Pk+1]
    w2 = wi1 * wj1 * wk0

    B3 = A[Pi, Pj+1, Pk]
    w3 = wi1 * wj0 * wk0

    B4 = A[Pi, Pj+1, Pk+1]
    w4 = wi1 * wj0 * wk0

    B5 = A[Pi+1, Pj, Pk]
    w5 = wi0 * wj1 * wk1

    B6 = A[Pi+1, Pj, Pk+1]
    w6 = wi0 * wj1 * wk0

    B7 = A[Pi+1, Pj+1, Pk]
    w7 = wi0 * wj0 * wk1

    B8 = A[Pi+1, Pj+1, Pk+1]
    w8 = wi0 * wj0 * wk0

    B = B1 * w1.unsqueeze(2).expand_as(B1) \
      + B2 * w2.unsqueeze(2).expand_as(B2) \
      + B3 * w3.unsqueeze(2).expand_as(B3) \
      + B4 * w4.unsqueeze(2).expand_as(B4) \
      + B5 * w5.unsqueeze(2).expand_as(B5) \
      + B6 * w6.unsqueeze(2).expand_as(B6) \
      + B7 * w7.unsqueeze(2).expand_as(B7) \
      + B8 * w8.unsqueeze(2).expand_as(B8)

    return B
