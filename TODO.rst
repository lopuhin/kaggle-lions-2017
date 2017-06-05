Prediction scale:
- check prediction quality on different scales
- 0.5 works best so far, 0.75 with sums is very poor, blobs a tiny bit better but still bad

Local validation:
- try to change validation scale randomly

Regression:
- visualize count predictions, or at least check them in the notebook
- try to vary patch size (40 used to perform worse)
- tune blobs generation once again?
- any other ideas about how to predict the sea lion count?

Training data:
- try a stratified split
- check labels on more training images (to catch bad orientation)
- rotate imagees https://www.kaggle.com/threeplusone/sea-lion-coordinates/comments/comments#186374

UNet training:
- lr schedule
- UNet hyperparameters:
    - filters_base
    - depth (via filter_factors)
    - 4x/8x pool on the last layer
- SGD
- why doesn't dice work? Try harder: oversampling, larger targets
- monitor regression loss (once validation is fixed)
- try to separate close lions to count them (predict distance)
- try class weight

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
