submit:
Sun:
- moar moar depth (done, good)
- moar depth with mid-14 (done, ~same)
- average mid-14 (done, ~ok)
Mon:
- make_submission patch 60 (done, good)
- fold2
- average masks?
- features from different masks?
- ?
Tue:
- ep13 more overlap?
- ep13 average???
- mega average - robust submission
- ?
- backup

TODO:

ws:
- train fold 3 with a different scale aug: 0.9 -- 1.5 (now)
- submit fold 2 (Mon)

1080:
- predict fold2 (ready Monday morning)
- predict new aug (ready Tue morning???)
- predict ep13 masks with more overlap (ready mid-Tue)

AWS:
- remove volumes and AMI

make_submission:
- average different masks
- an option to predict without rounding
- try patch size 60
- use features from several different masks
- more features from mask: more thresholds, ???
- less features from mask

classification:
- add scale augmentation (0.8 -- 1.25 for a start)
- vary patch size

UNet:
- Larger markers for cls 0 an 1 (12 and 10?)
- Try more conv layers in UNet

Try to reduce scale augmentation during training? say 0.8 -- 1.25 - bad?

make_submission:
- use features from several different masks
- average different masks
- try a second level model that accepts class predictions
- do something different for each class?
  - use different features
  - take from different predictions
  - multiply by some constant, validate on the LB?

Last submission:
Average several variations of the best LB submission
Average several mid-14 submissions

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
