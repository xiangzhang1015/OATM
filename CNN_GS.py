from CNN import *
import time

start_time = time.time()
for lr in [0.001, 0.003, 0.005]:
    for lam in [0.004, 0.008, 0.012]:
        for n_l in [1, 2, 3]:
            for n_n in [64, 128, 192]:
                cnn_run(lr, lam, n_l, n_n)

print("Running time of the optimization is ", time.time() - start_time)
