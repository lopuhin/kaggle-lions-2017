- don't use black areas in TrainDotted - they cover unlabeled lions
  that would be false negatives during training
- recheck missing labels during training - will they still be there?
- stratified split by count - not sure?

prediction:
- finish current approach with integrating
- try thresholds + counting connected components

UNet training:
- better way to monitor progress - loss only on other classes?
- try dice loss (maybe modify it - pos/neg weights, square size)
- try to separate close lions to count them (predict distance)
- determine image scale, rescale!

performance:
- prediction speed - make more shallow? maybe make FCN UNet?
- multi-threaded data loader to train at full speed on full dataset
  (maybe training on 800 would be good enough?)

other ideas:
- check SOTA detection models - SSD, any other?

later:
- smart rounding (if it's required at all)
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

