import time

from CNN import *
from bayes_opt import BayesianOptimization


# define opt function
def cnn_bo(lr, lam, n_l,n_n):
    acc = cnn_run(lr=lr, lam=lam, n_l=int(n_l), n_n=int(n_n))
    return acc


bo = BayesianOptimization(cnn_bo,
                                {
                                 "lr": (0.001, 0.005),
                                 "lam": (0.004, 0.012),
                                 "n_l": (1, 3),
                                 "n_n": (64, 192),
                                })

num_iter = 7
init_points = 2
start_time = time.time()
bo.maximize(init_points=init_points,n_iter=num_iter)
print("Running time of the optimization is ", time.time() - start_time)
print(bo.max)
