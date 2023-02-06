#!/bin/bash
mamba update mamba -y
mamba create -n job-env -y python=3.9
mamba install -n job-env -y mne kfp joblib pandas scikit-learn skorch h5py seaborn matplotlib qt libmamba -c conda-forge 
mamba install -n job-env -y pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
#mamba install -n job-env -y ipykernel -c anaconda 
#python -m ipykernel install --user --name=job-env
#mamba env export -n job-env -f /home/jovyan/environment_new.yml



#
mamba create -n job-env -y python=3.9 mne kfp joblib pandas scikit-learn skorch h5py seaborn matplotlib qt libmamba pytorch pytorch-cuda=11.6 -c pytorch -c nvidia -c conda-forge
