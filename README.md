# Orthogonal Array Tuning: An Efficient Hyper-parameter Tuning Method on Deep Learning
## Title: Deep Neural Network Hyperparameter Optimization with Orthogonal Array Tuning

**PDF: [ICONIP2019](https://link.springer.com/chapter/10.1007/978-3-030-36808-1_31), [arXiv](https://arxiv.org/abs/1907.13359)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu),  Xiaocong Chen, [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Chang Ge, Manqing Dong**

## Overview
The proposed OATM (Orthogonal Array Tuning Method) adopts the orthogonal array to extract the most representative and balanced combinations from the whole set of possible combinations. In detail, the OATM manner is proposed based on Taguchi Approach. The OATM is a highly fractional orthogonal design method that is based on a design matrix and allows the user to consider a selected subset of combinations of multiple factors at multiple levels. Additionally, the OATMis balanced to ensure that all possible values of all hyper-parameters are considered equally.

## CNN and RNN
- The code for CNN is in CNN.py and the code for RNN is in RNN.py
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

## Optimization

- We provide running files to run different optimization methods with our CNN and RNN integrated.
- In order to run Bayesian Optimization for CNN and RNN, simply run CNN_BO.py and RNN_BO.py
- In order to run Grid search, simply run CNN_GS.py and RNN_GS.py


## Citing
If you find our work useful for your research, please consider citing this paper:

    @inproceedings{zhang2019deep,
      title={Deep neural network hyperparameter optimization with orthogonal array tuning},
      author={Zhang, Xiang and Chen, Xiaocong and Yao, Lina and Ge, Chang and Dong, Manqing},
      booktitle={International conference on neural information processing},
      pages={287--295},
      year={2019},
      organization={Springer}
    }


## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
