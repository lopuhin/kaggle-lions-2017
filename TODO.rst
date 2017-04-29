- recheck missing labels during training - will they still be there?
- stratified split by count - not sure?

UNet training:
- better way to monitor progress - loss only on other classes?
- try dice loss (maybe modify it - pos/neg weights, square size)
- try to separate close lions to count them (predict distance)
- determine image scale, rescale!

performance:
- generate features in parallel with test predictions
- prediction speed - make more shallow? maybe make FCN UNet?
- fit predictions into uint8, downscale 4x in UNet
- multi-threaded data loader to train at full speed on full dataset
  (maybe training on 800 would be good enough?)

other ideas:
- check SOTA detection models - SSD, any other?

later:
- smart rounding
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

