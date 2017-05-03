Local validation vs. LB:
- need better features (lasso weights show that)
  and a "local" final classifier (per-blob or per-patch)

Training data:
- still some missing labels during training - more mislabeled images?
- stratified split by count - not sure?
- determine image scale, rescale!

UNet training:
- better way to monitor progress - loss only on other classes? monitor dice loss?
- try dice loss (maybe modify it - pos/neg weights, square size)
- try to separate close lions to count them (predict distance)

Performance:
- do not predict black areas on test
- downscale 4x or 2x in UNet
- generate features in parallel with test predictions
- try more shallow UNet? maybe make an FCN UNet?

Test data:
- test predictions viewer
- check 41 - funny black shape
- check 53 - large scale

Other ideas:
- check SOTA detection models - SSD, any other?

Later:
- smart rounding
- check reports here: https://www.afsc.noaa.gov/nmml/alaska/
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

