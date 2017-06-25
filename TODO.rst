TODO:

ws:
- continue training AWS model,
$ ./unet.py runs/unet-stratified-scale-0.8-1.6-oversample0.2-fold2 --stratified --fold 2 --min-scale 0.8 --max-scale 1.6 --oversample 0.2 --workers 0
- more hyperparameter tuning

1080:
- predict fold2

AWS:
- remove volumes and AMI

classification:
- add scale augmentation (0.8 -- 1.25 for a start)
- vary patch size

UNet:
- Larger markers for cls 0 an 1 (12 and 10?)
- Try more conv layers in UNet

Try to reduce scale augmentation during training? say 0.8 -- 1.25 - bad?

make_submission:
- try a second level model that accepts class predictions
- do something different for each class?
  - use different features
  - take from different predictions
  - multiply by some constant, validate on the LB?

Last submission:
Maybe have at least one blobs submission
Try to average predictions from several models
More overlap on UNet prediction: could make sense

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
