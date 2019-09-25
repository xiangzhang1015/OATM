# Orthogonal Array Tuning: An Efficient Hyper-parameter Tuning Method on Deep Learning

In this repo, we provide the code and the datasets that we use to conduct this research topic. 

## Notice, BELLOW is only the example citation I copied from your homepage, when you uploaded to the real git repo, please change it to the actual citation 
Xiang Zhang, Xiaocong Chen, Lina Yao, Chang Ge, and Manqing Dong. Deep Neural Network Hyperparameter Optimization with Orthogonal Array Tuning. The 26th Interna tional Conference On Neural Information Processing (ICONIP 2019). Sydney, Australia, December 12-15, 2019. Retrieved from: https://arxiv.org/abs/1907.13359


### CNN and RNN

- The code for CNN is in [CNN.py](https://github.com/) and the code for RNN is in [RNN.py](https://github.com/)
- In this code, the default data is EEG data, the other two data sets are commented. In order to run other datasets, simply comment the "EEG" part and uncomment the wanted datasets
```python
# ---------------- EEG data ----------------
# default running code here
feature = sc.loadmat("S1_nolabel6.mat")
all = feature['S1_nolabel6']

# --------------- RFID data --------------------
# commented code here
# ......
# ......
# # -------------- PAMAP2 data ------------------------
# .......
# .....
```

### Optimization

- We provide running files to run different optimization methods with our CNN and RNN integrated.
- In order to run Bayesian Optimization for CNN and RNN, simply run CNN_BO.py and RNN_BO.py
- In order to run Grid search, simply run CNN_GS.py and RNN_GS.py
