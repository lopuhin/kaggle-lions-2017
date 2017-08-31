NOAA Fisheries Steller Sea Lion Population Count
================================================

This is a 2nd place solution for
`Kaggle NOAA Fisheries Steller Sea Lion Population Count <https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count>`_.

A short overview of the solution on kaggle forums:
https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35422

Installation
------------

Install required packages (Ubuntu 16.04 or above, CUDA is assumed
to be already installed)::

    sudo apt install \
        gcc make python3-venv python3-pip python3-dev python3-tk libgeos-dev

Make a virtual environment and install python packages::

    python3.5 -m venv venv
    . venv/bin/activate
    pip install -U pip wheel

Install pytorch ``0.1.12.post2``: go to http://pytorch.org, select the version
according to your CUDA and python version, and install it. For example,
for CUDA 8.0 on Python 3.5 and Ubuntu, the command would be::

    pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl

Install the rest of python packages::

    pip install -r requirements.txt

Download training and testing data (``Test``, ``Train``, ``TrainDotted``)
and place it in ``./data`` folder so it looks like this::

    data
    ├── coords-threeplusone-v0.4.csv
    ├── MismatchedTrainImages.txt
    ├── sample_submission.csv
    ├── Test
    ├── Train
    └── TrainDotted


Training and making a submission
--------------------------------

Make a directory ``_runs``::

    mkdir _runs

Train UNet (this takes about 20 hours and needs 8GB of GPU memory)::

    ./unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 \
        --stratified \
        --batch-size 32 \
        --min-scale 0.8 --max-scale 1.6 \
        --n-epochs 13 \
        --oversample 0.2

**Note:** it may be that 8 GB is slightly not enough and the code may crash
due to GPU memory error after running for one epoch. In this case, run training
in a bash loop (the model is saved and restored successfully).

Result of this step (``best-model.pt``) is provided at
``models/unet-stratified-scale-0.8-1.6-oversample0.2/`` in the archive.

Make predictions for validation set (and all other images not used due to ``--limit``,
if ``--predict_all_valid`` option is used instead of ``--predict_valid``, like below),
also don't forget to pass all the other params from training
(this should take less than an hour)::

    ./unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 \
        --stratified \
        --batch-size 32 \
        --min-scale 0.8 --max-scale 1.6 \
        --n-epochs 13 --oversample 0.2 \
        --mode predict_all_valid

Train a regression model on this predictions (takes less than 10 minutes)::

    ./make_submission.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 train

Result of this step (``regressor.joblib``) is provided at
``models/unet-stratified-scale-0.8-1.6-oversample0.2/`` in the archive.

Now you need to predict all test (this takes about 12 hours)::

    ./unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 \
        --stratified \
        --batch-size 32 \
        --min-scale 0.8 --max-scale 1.6 \
        --n-epochs 13 --oversample 0.2 \
        --mode predict_test

Now make submission with (takes a few hours)::

    ./make_submission.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 predict

Submission file will be in
``_runs/unet-stratified-scale-0.8-1.6-oversample0.2/unet-stratified-scale-0.8-1.6-oversample0.2.csv``
