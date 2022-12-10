Install Skater
================

Dependencies
~~~~~~~~~~~~~~~~
Skater relies on 

- scikit-learn>=0.18,
- pandas>=0.19,
- ds-lime>=0.1.1.21(datascience.com forked version of LIME),
- requests,
- multiprocess,
- joblib==0.11,
- dill>=0.2.6,
- rpy2==2.9.1; python_version>"3.0",
- numpy
- with v1.1.0 there are additional dependencies on R related binaries(setup.sh)

External/Optional dependencies

- Plotting functionality requires matplotlib>=2.1.0
- tensorflow>=1.4.0
- keras>=2.0.8

pip
~~~~
We recommended that you use a virtual environment to ensure your system is not modified by an installation (virtualenv, conda environment).
Detailed instruction could be found at `install conda <https://conda.io/docs/user-guide/install/linux.html>`_
::
    Quick summary to install conda and setup the python environment(recommended steps for using python3.x)

    1. wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    2. bash miniconda.sh -b -p $HOME/miniconda
    3. export PATH="$HOME/miniconda/bin:$PATH"
    4. conda config --set always_yes yes --set changeps1 no
    5. conda info -a

Managing conda virtual environment. For details, check `here <https://conda.io/docs/user-guide/tasks/manage-environments.html#activating-an-environment>`_
::

    1. create: conda create -n skater-test python=3.6
    2. activate: source activate skater-test
    3. deactivate: source deactivate

Install
::

    Option 1: without rule lists and without deepinterpreter
    pip install -U skater

    Option 2: without rule lists and with deepinterpreter:
    1. Ubuntu: pip3 install --upgrade tensorflow (follow instructions at https://www.tensorflow.org/install/ for details and          best practices)
    2. sudo pip install keras
    3. pip install -U skater
    
    Option 3: For everything included
    1. conda install gxx_linux-64
    2. Ubuntu: pip3 install --upgrade tensorflow (follow instructions https://www.tensorflow.org/install/ for 
       details and best practices)
    3. sudo pip install keras
    4. sudo pip install -U --no-deps --force-reinstall --install-option="--rl=True" skater==1.1.1b1
    

conda
~~~~~~
Skater is available in the conda-forge channel with builds for Linux and OS X. 
The latest version of the library is not updated on conda, `#223 <https://github.com/datascienceinc/Skater/issues/223>`_)
::

    conda install -c conda-forge Skater


testing
~~~~~~~~
::

    python -c "from skater.tests.all_tests import run_tests; run_tests()"
