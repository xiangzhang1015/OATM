from RNN import *
from bayes_opt import BayesianOptimization
import time


# define optimization function
# input hyperparameters needed to be adjust
def rnn_bo(lr, lam, n_layers, nodes):
    acc = rnn_run(lr=lr, lam=lam, n_layers=int(n_layers), nodes=int(nodes))
    return acc


# specify the values range of required hyperparameters
bo = BayesianOptimization(rnn_bo,
                            {
                                "lr": (0.005, 0.15),
                                "lam": (0.004, 0.012),
                                "n_layers": (4, 6),
                                "nodes": (32, 96)
                            })

num_iter = 7
init_points = 2
start_time = time.time()
bo.maximize(init_points=init_points, n_iter=num_iter)
print("Running time of the optimization is ", time.time() - start_time)
