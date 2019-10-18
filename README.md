# Optimal Sparse Decision Trees (OSDT)

### Dependencies

* [gmp](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library)
* [mpfr](http://www.mpfr.org/) (GNU MPFR Library for multiple-precision floating-point computations; depends on gmp)
* [libmpc](http://www.multiprecision.org/) (GNU MPC for arbitrarily high precision and correct rounding; depends on gmp and mpfr)
* [gmpy2](https://pypi.org/project/gmpy2/#files) (GMP/MPIR, MPFR, and MPC interface to Python 2.6+ and 3.x)

### Main function
The main function is the `bbound()` function in `osdt.py`.

### Arguments

**[x]** The features of the training data.

**[y]** The labels of the training data.

**[lamb]** The regularization parameter `lambda` of the objective function.

**[prior_metric]** The scheduling policy.

* Use `curiosity` to prioritize by curiosity (see *Section 5* of our paper).
* Use `bound` to prioritize by the lower bound.
* Use `objective` to prioritize by the objective.
* Use `FIFO` for first-in-first-out search.

**[MAXDEPTH]** Maximum depth of the tree. Default value is `float('Inf')`.

**[MAX_NLEAVES]** Maximum number of leaves of the tree. Default value is `float('Inf')`.

**[niter]** Maximum number of tree evaluations. Default value is `float('Inf')`.

**[logon]** Record relevant trees and values during the execution. Default if `False`.

**[support]** Turn on `Lower bound on leaf support`. Default is `True`.

**[incre_support]** Turn on `Lower bound on incremental classification accuracy`. Default is `True`.

**[accu_support]** Turn on `Lower bound on classification accuracy`. Default is `True`.

**[equiv_points]** Turn on `Equivalent points bound`. Default is `True`.

**[lookahead]** Turn on `Lookahead bound`. Default is `True`.

**[lenbound]** Turn on `Prefix-specific upper bound on number of leaves`. Default is `True`.

**[timelimit]** Time limit on the running time. Default is `True`.

**[init_cart]** Initialize with CART. Default is `True`.

### Example test code

We provide our test code in `test_accuracy.py`.

### Dataset

See `data/preprocessed/`.

We used 7 datasets: Five of them are from the UCI Machine Learning Repository (tic-tac-toc, car evaluation, monk1, monk2, monk3). 
The other two datasets are the ProPublica recidivism data set and the Fair Isaac (FICO) credit risk datasets. 
We predict which individuals are arrested within two years of release (${N = 7,215}$) on the recidivism data set and whether an individual will default on a loan for the FICO dataset. 