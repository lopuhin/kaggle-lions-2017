Local validation vs. LB:
- make submission

Regression:
- try more features
- try concated xs
- try smaller patch size
- check confusion

Training data:
- still some missing labels during training - more mislabeled images?
- determine image scale, rescale!

UNet training:
- why doesn't dice work? Try harder: oversampling, larger targets
- monitor regression loss
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
- check window classification one again - see car counting paper

Later:
- smart rounding
- final regression on sums
- check reports here: https://www.afsc.noaa.gov/nmml/alaska/
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups
