make_submission:
- more estimators, tune params - train xgboost with validation, use that tuned
  model for prediction

Try jaccard again with more oversampling

More overlap on UNet prediction
Try slightly different test scale (2.2? 1.8?)
Try to reduce scale augmentation during training?

Try predicting scale
Try predicting borders - check original UNet paper

Prediction scale:
- train 1.0 -- 2 scale + lr schedule + stratified,
  predict test at 0.5 with a small validation variation -- bad :(

Regression:
- visualize count predictions, or at least check them in the notebook
- try to vary patch size (40 used to perform worse)
- tune blobs generation once again?
- any other ideas about how to predict the sea lion count?

Training data:
- check labels on more training images (to catch bad orientation)
- rotate imagees https://www.kaggle.com/threeplusone/sea-lion-coordinates/comments/comments#186374

UNet training:
- SGD
- UNet hyperparameters:
    - larger patch size
    - filters_base (not much diff?)
    - depth via filter_factors (not much diff?)
    - 4x/8x pool on the last layer (not much diff?)
- try class weight, say 0.1 for no lion (bad)
- why doesn't dice work? Try harder: oversampling, larger targets
- monitor regression loss once validation is fixed
- try to separate close lions to count them, predict distance?

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
- do dot regression (like SSD but without bounding boxes?)
- check window classification once again - see car counting paper

Later:
- smart rounding
- final regression on sums
- check reports here: https://www.afsc.noaa.gov/nmml/alaska/
- more features for count prediction:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups
