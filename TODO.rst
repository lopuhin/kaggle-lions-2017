- recheck missing labels during training - will they still be there?
- stratified split by count - not sure?

UNet training:
- better way to monitor progress - loss only on other classes?
- try dice loss (maybe modify it - pos/neg weights, square size)
- try to separate close lions to count them (predict distance)
- determine image scale, rescale!

performance:
- downscale 4x or 2x in UNet
- generate features in parallel with test predictions
- try more shallow UNet? maybe make an FCN UNet?

other ideas:
- check SOTA detection models - SSD, any other?

later:
- smart rounding
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

