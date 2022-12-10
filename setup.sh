#!/bin/sh
# Requirements: R >= 3.4.0
source ~/.bashrc # just as a pre-caution, to make sure all system paths are updated post library installations

# Install python related dev tools as per the python version. This is needed for gcc installation
ver=$(python -V 2>&1 | sed 's/. *\([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" = "36" ]; then
    echo "python 3.6.x detected"
    sudo apt-get install python3.6-dev
    exit 0
elif [ "$ver" = "35" ]; then
    echo "python 3.5.x detected"
    sudo apt-get install python3.5-dev
    exit 0
elif [ "$ver" = "27" ]; then
    echo "This script requires python 2.7 or greater"
    sudo apt-get install libffi-dev
fi

if [ "$1" = "mac" ]
then
    brew install r
    brew install r-cran-rcpp
    # installing amp
    brew install libgmp3-dev
elif [ "$1" = "linux-ubuntu" ]
then
    # installs R base as well
    # decent info on installing R manually could be found here at the link mentioned below,
    # https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-16-04-2
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    sudo add-apt-repository -y ppa:marutter/rrutter
    sudo apt-get update
    sudo apt-get install libgmp3-dev
    sudo apt-get install gsl-bin
    sudo apt-get install libgsl2
    sudo apt-get install libgsl0-dev
    sudo apt-get -y install r-base
    sudo apt-get install r-cran-rcpp
elif [ "$1" = "linux-rpm" ]
then
    sudo yum -y install r-base
    sudo yum install r-cran-rcpp
    sudo yum install libgmp3-dev
fi

echo "installing rpy2"
sudo python -m pip install rpy2==2.9.4

# downloads the required R packages locally in the same directory as setup.py
wget https://cran.r-project.org/src/contrib/Archive/Rcpp/Rcpp_0.12.16.tar.gz
sudo R CMD INSTALL Rcpp_0.12.16.tar.gz
wget https://cran.r-project.org/src/contrib/Archive/arules/arules_1.5-5.tar.gz
sudo R CMD INSTALL arules_1.5-5.tar.gz
wget https://cran.r-project.org/src/contrib/sbrl_1.2.tar.gz
sudo R CMD INSTALL sbrl_1.2.tar.gz