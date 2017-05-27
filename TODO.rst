Local validation:
- try to change scale randomly

Regression:
- try to vary patch size (40 used to perform worse)
- make blobs great again? currently sum is the best feature
- any other ideas about how to prediction sea lion count

Training data:
- switch to coords-threeplusone-v0.4.csv
- check labels on more training images (to catch bad orientation)

UNet training:
- lr schedule
- UNet hyperparameters
- SGD
- why doesn't dice work? Try harder: oversampling, larger targets
- monitor regression loss
- try to separate close lions to count them (predict distance)
- class weight

Performance:
- downscale 4x or 2x in UNet output
- do not predict black areas on test
- generate features in parallel with test predictions
- try more shallow UNet?
- make an FCN UNet?

Test data:
- test predictions viewer
- check 41 - funny black shape
- check 53 - large scale

Other ideas:
- check SOTA detection models - SSD, any other?
- check window classification once again - see car counting paper

Later:
- smart rounding
- final regression on sums
- check reports here: https://www.afsc.noaa.gov/nmml/alaska/
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups
