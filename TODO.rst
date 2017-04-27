- check missing labels during training - are they missing in dotted images?
- check MismatchedTrainImages.txt - are they in coords?

prediction:
- finish current approach with integrating
- try thresholds + counting connected components

UNet training:
- better way to monitor progress - loss only on other classes?
- try dice loss (maybe modify it - pos/neg weights, square size)
- try to separate close lions to count them
- determine image scale, rescale

performance:
- prediction speed! parallelize, maybe make FCN UNet
- multi-threaded data loader to train at full speed on full dataset

other ideas:
- check SOTA detection models - SSD, any other?

later:
- check quality of coordinates carefully on all images
- predict one class (pups) based on other classes:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

