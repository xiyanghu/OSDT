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

model = OSDT()
model.fit(x_train, y_train)
prediction, accuracy = model.predict(x_test, y_test)
```

### Dependencies

* [gmp](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library)
* [mpfr](http://www.mpfr.org/) (GNU MPFR Library for multiple-precision floating-point computations; depends on gmp)
* [libmpc](http://www.multiprecision.org/) (GNU MPC for arbitrarily high precision and correct rounding; depends on gmp and mpfr)
* [gmpy2](https://pypi.org/project/gmpy2/#files) (GMP/MPIR, MPFR, and MPC interface to Python 2.6+ and 3.x)


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
