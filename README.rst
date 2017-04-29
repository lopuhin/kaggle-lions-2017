NOAA Fisheries Steller Sea Lion Population Count
================================================

`Kaggle <https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count>`_


Training and making a submission
--------------------------------

Train UNet::

    ./unet.py runs/unet-limit500 --limit 500

Make predictions for validation set (and all other images not used due to ``--limit``,
if ``--predict_all_valid`` option is used instead of ``--predict_valid``, like below),
also don't forget to pass all the other params from training::

    ./unet.py runs/unet-limit500 --limit 500 --mode predict_all_valid

Train a regression model on this predictions::

    ./make_submission.py runs/unet-limit500 train

Now you need to predict all test (that takes a lot of time)::

    ./unet.py runs/unet-limit500 --limit 500 --mode predict_test

Now make submission with::

    ./make_submission.py runs/unet-limit500 predict

