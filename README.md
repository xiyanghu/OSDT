# Optimal Sparse Decision Trees (OSDT)

![GitHub Repo stars](https://img.shields.io/github/stars/xiyanghu/OSDT?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/hu_xiyang?style=social)
![License](https://img.shields.io/github/license/xiyanghu/OSDT?color=critical)
[![arXiv](https://img.shields.io/badge/arXiv-1904.12847-b31b1b.svg?style=flat)](https://arxiv.org/abs/1904.12847)

This accompanies the paper, ["Optimal Sparse Decision Trees"](https://arxiv.org/abs/1904.12847) by Xiyang Hu,
Cynthia Rudin, and Margo Seltzer.

It appeared in the [2019 NeurIPS conference](https://nips.cc/Conferences/2019)

* [:movie_camera: Overview video](https://youtu.be/UMjMQaH508M)
* [:newspaper: NeurIPS poster](doc/OSDT_NIPS_Poster.pdf)
* [:notebook_with_decorative_cover: NeurIPS slides](doc/NeurIPSSlides.pdf)

### Use OSDT

```python
from osdt import OSDT

# initilize an OSDT object
model = OSDT()
# fit the model
model.fit(x_train, y_train)
# make prediction and get the prediction accuracy
prediction, accuracy = model.predict(x_test, y_test)
# make prediction only
prediction = model.predict(x_test)
```

### Documentation

All code are in the `./src` folder. The `OSDT` class is in the `osdt.py` file.

---

```python
CLASS osdt.OSDT(lamb=0.1, prior_metric="curiosity", MAXDEPTH=float('Inf'), MAX_NLEAVES=float('Inf'), niter=float('Inf'),
                logon=False, support=True, incre_support=True, accu_support=True, equiv_points=True,
                lookahead=True, lenbound=True, R_c0=1, timelimit=float('Inf'), init_cart=True,
                saveTree=False, readTree=False)
```

<details><summary> <b>PARAMETERS</b>: </summary>
<p>
 
 * **lamb** : float, optional (default=0.1)\
     The regularization parameter lambda of the objective function.
 prior_metric : {'objective', 'bound', 'curiosity', 'entropy', 'gini', 'FIFO'}, optional (default='curiosity')
     The scheduling policy used to determine the priority of leaves:
     - 'objective' will use the objective function
     - 'bound' will used the lower bound
     - 'curiosity' will use the curiosity
     - 'entropy' will use the entropy
     - 'gini' will use the GINI value
     - 'FIFO' will use first in first out
 * **MAXDEPTH** : int, optional (default=float('Inf'))\
     Maximum depth of the tree.
 * **MAX_NLEAVES** : int, optional (default=float('Inf'))\
     Maximum number of leaves of the tree.
 * **niter** : int, optional (default=float('Inf'))\
     Maximum number of tree evaluations.
 * **logon** : bool, optional (default=False)\
     Record relevant trees and values during the execution.
 * **support** : bool, optional (default=True)\
     Turn on Lower bound on leaf support.
 * **incre_support** : bool, optional (default=True)\
     Turn on Lower bound on incremental classification accuracy.
 * **accu_support** : bool, optional (default=True)\
     Turn on Lower bound on classification accuracy.
 * **equiv_points** : bool, optional (default=True)\
     Turn on Equivalent points bound.
 * **lookahead** : bool, optional (default=True)\
     Turn on Lookahead bound.
 * **lenbound** : bool, optional (default=True)\
     Turn on Prefix-specific upper bound on number of leaves.
 * **R_c0** : float, optional (default=1)\
     The initial risk.
 * **timelimit** : int, optional (default=float('Inf'))\
     Time limit on the running time. Default is True.
 * **init_cart** : bool, optional (default=True)\
     Initialize with CART.
 * **saveTree** : bool, optional (default=False)\
     Save the tree.
 * **readTree** : bool, optional (default=False)\
     Read Tree from the preserved one, and only explore the children of the preserved one.

</p>
</details>

---

> **fit**(x, y)

&nbsp;&nbsp;&nbsp; Fit the model with input data.

&nbsp;&nbsp;&nbsp; **PARAMETERS**:
* **x** : ndarray of shape (ndata, nfeature)\
    The features of the data.
* **y** : ndarray of shape (ndata,), optional\
    The true labels of the data.

&nbsp;&nbsp;&nbsp; **RETURNS**:
* **self** : fitted OSDT model.

> **predict**(x, y=None)

&nbsp;&nbsp;&nbsp; Predict if a particular sample is an outlier or not.

&nbsp;&nbsp;&nbsp; **PARAMETERS**:
* **x** : ndarray of shape (ndata, nfeature)\
    The features of the data.
* **y** : ndarray of shape (ndata,), optional (default=None)\
    The true labels of the data.

&nbsp;&nbsp;&nbsp; **RETURNS**:
* **prediction** : ndarray of shape (ndata,)\
    The features of the training data.
* **accuracy** : float, optional\
    If true label y is provided, output the accuracy of the prediction.

---

### Installation

```shell
git clone https://github.com/xiyanghu/OSDT.git
cd OSDT
conda env create -f environment.yml
conda activate osdt
```

#### Dependencies

* [gmp](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library)
* [mpfr](http://www.mpfr.org/) (GNU MPFR Library for multiple-precision floating-point computations; depends on gmp)
* [libmpc](http://www.multiprecision.org/) (GNU MPC for arbitrarily high precision and correct rounding; depends on gmp and mpfr)
* [gmpy2](https://pypi.org/project/gmpy2/#files) (GMP/MPIR, MPFR, and MPC interface to Python 2.6+ and 3.x)
* See [environment.yml](environment.yml)

<!---
1. Install GMP
   * Run Command `sudo apt install libgmp3-dev`(Ubuntu) OR `brew install gmp`(MacOS) 
   * If the command above does not work, try manual Installation:
      * Download `gmp-6.2.1.tar.xz` from [gmplib.org](https://gmplib.org/)
      * Run command `tar -jvxf gmp-6.2.1.tar.xz`
      * Run command `cd gmp-6.2.1`
      * Run command `./configure`
      * Run command `make`
      * Run command `make check`
      * Run command `sudo make install`
2. Install MPFR
   * Run command `sudo apt install libmpfr-dev`(Ubuntu) OR `brew install mpfr`(MacOS)  
3. Install libmpc
   * Run command `sudo apt install libmpc-dev`(Ubuntu) OR `brew install libmpc`(MacOS)  
4. Install gmpy2
   * Run command `pip install gmpy2`
-->

### Datasets

See `data/preprocessed/`.

We used 7 datasets: Five of them are from the UCI Machine Learning Repository (tic-tac-toc, car evaluation, monk1, monk2, monk3). 
The other two datasets are the ProPublica recidivism data set and the Fair Isaac (FICO) credit risk datasets. 
We predict which individuals are arrested within two years of release (`{N = 7,215}`) on the recidivism data set and whether an individual will default on a loan for the FICO dataset. 
* [Tic-Tac-Toc](https://archive.ics.uci.edu/ml/datasets/tic-tac-toe+Endgame)
* [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/car+evaluation)
* [MONK's](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems)
* [ProPublica](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)
* [FICO](https://community.fico.com/s/explainable-machine-learning-challenge)


### Example test code

We provide our test code in `test_accuracy.py`.

### Citing OSDT

[OSDT paper](<https://arxiv.org/abs/1904.12847>) is published in
*Neural Information Processing Systems (NeurIPS) 2019*.
If you use OSDT in a scientific publication, we would appreciate
citations to the following paper:

    @inproceedings{NEURIPS2019_ac52c626,
     author = {Hu, Xiyang and Rudin, Cynthia and Seltzer, Margo},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
     pages = {7265-7273},
     publisher = {Curran Associates, Inc.},
     title = {Optimal Sparse Decision Trees},
     url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/ac52c626afc10d4075708ac4c778ddfc-Paper.pdf},
     volume = {32},
     year = {2019}
    }


or:

    Hu, X., Rudin, C., and Seltzer, M. (2019). Optimal sparse decision trees. In Advances in Neural Information Processing Systems, pp. 7265–7273.
