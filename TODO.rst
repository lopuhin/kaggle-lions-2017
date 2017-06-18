make_submission:
- more estimators, tune params - train xgboost with validation, use that tuned
  model for prediction

Try to reduce scale augmentation during training? say 0.8 -- 1.25
Try to sample random lion across all images
Try larger patch size (384 x 384)
Try slightly different test scale (0.55?)
Try to train on 1.5x and predict on 0.75x?
Try stratified again?

Last submission:
More overlap on UNet prediction

Crazy:
Try predicting scale
Try predicting borders - check original UNet paper

Try jaccard again with more oversampling (looks worse)

UNet training:
- SGD (looks worse)
- UNet hyperparameters:
    - larger patch size
    - filters_base (not much diff?)
    - depth via filter_factors (not much diff?)
    - 4x/8x pool on the last layer (not much diff?)

Performance:
- make an FCN UNet?

Later:
- smart rounding
- check reports here: https://www.afsc.noaa.gov/nmml/alaska/
