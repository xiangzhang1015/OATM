from RNN import *
import time

start_time = time.time()
for lr in [0.005, 0.1, 0.15]:
    for lam in [0.004, 0.008, 0.012]:
        for n_layers in [4, 5, 6]:
            for nodes in [32, 64, 96]:
                rnn_run(lr, lam, n_layers, nodes)

print("Running time of the optimization is ", time.time() - start_time)