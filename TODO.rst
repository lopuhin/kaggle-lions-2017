Local validation vs. LB:
- reasons:
  - bug in how predictions are generated? re-run with correct step
  - something is different between train and test?
  - scale seems to be very different! make predictions with half scale
- check test predictions
- why is mean for cls-2 (adult_females) much higher than what is used in baseline submissions?
- try training on another fold - check why is validation so much worse?

Regression:
- try smaller patch size

Training data:
- https://www.kaggle.com/threeplusone/sea-lion-coordinates/comments/comments#182401
- still some missing labels during training - more mislabeled images?
- determine image scale, rescale!

UNet training:
- why doesn't dice work? Try harder: oversampling, larger targets
- monitor regression loss
- try to separate close lions to count them (predict distance)
- scale augmentation
- class weight
- UNet hyperparameters

./make_submission.py runs/unet-limit800-clean-lr0.0004/ predict --new-features

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
