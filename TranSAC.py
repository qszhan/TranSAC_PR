import numpy as np 

def continuous_entropy(z, eps, order):
    m, d = z.shape
    mu = (m + d) / 2
    lamda = d / (m*eps*eps)
    c =  z @ z.transpose()
    c = lamda*c
    power_matrix = c
    sum_matrix = np.zeros_like(power_matrix)
    for k in range(1, order+1):
        if k > 1:
            power_matrix = np.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else:
            sum_matrix -= power_matrix / k
    trace = np.trace(mu*sum_matrix)
    entropy = trace
    return entropy


def compute_TranSAC(tar_prob, T, args):
    T = T - np.mean(T, axis=0, keepdims=True)
    ht = continuous_entropy(T, eps=args.eps, order=args.order)
    K = tar_prob.shape[1]
    hzct = -np.sum(tar_prob * np.log(tar_prob.clip(min=1e-10)))   
    TranSAC = ht - hzct / k
    return TranSAC
 
