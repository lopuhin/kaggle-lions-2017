cls 0, patch mean 0.026, patch RMSE 0.143, image mean 5.74, image RMSE 3.77, baseline RMSE 6.49
cls 1, patch mean 0.020, patch RMSE 0.176, image mean 4.44, image RMSE 5.60, baseline RMSE 6.13
cls 2, patch mean 0.191, patch RMSE 0.555, image mean 42.66, image RMSE 20.47, baseline RMSE 63.12
cls 3, patch mean 0.090, patch RMSE 0.525, image mean 20.01, image RMSE 18.02, baseline RMSE 26.75
cls 4, patch mean 0.079, patch RMSE 0.365, image mean 17.57, image RMSE 15.99, baseline RMSE 41.49
ExtraTreesRegressor, XGBRegressor: mean patch RMSE 0.353, mean image RMSE 12.77, mean baseline RMSE 28.80

Local validation vs. LB:
- reasons:
  - leak?
  - bug in how predictions are generated?
  - something is different between train and test?
  - scale seems to be very different!
- check test predictions
- why is mean for cls-2 (adult_females) much higher than what is used in baseline submissions?
- try training on another fold
- try clipping

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
