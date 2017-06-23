TODO:

Try 192 patch
Larger markers for cls 0 an 1 (12 and 10?)
Try more conv layers in UNet

1080:
- predict 192 patch
- train & predict another fold with current best settings?

Try to reduce scale augmentation during training? say 0.8 -- 1.25 - bad?
Try slightly different test scale (0.55?)
Try to train on 1.5x and predict on 0.75x?


make_submission:
- try predicting with old features (substracting sums)
- try concat again
- try a second level model that accepts class predictions
- do something different for each class?
  - use different features
  - take from different predictions
  - multiply by some constant, validate on the LB?

Last submission:
Try to average predictions from several models
More overlap on UNet prediction: could make sense
Train/predict different folds


Crazy:
Try predicting scale
Try predicting borders - check original UNet paper
Add some vgg-like or resnet head, and make predictions 4x smaller - bad

Try jaccard again with more oversampling (looks worse)

UNet training:
- SGD (looks worse)
- UNet hyperparameters:
    - filters_base (not much diff?)
    - depth via filter_factors (not much diff?)
    - 4x/8x pool on the last layer (not much diff?)

Performance:
- make an FCN UNet?
