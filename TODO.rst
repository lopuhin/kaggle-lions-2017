- train on full dataset
- accuracy report, confusion matrix
- visualize detector predictions
- image scale - how much does it vary? how to estimate it?
  just label and add scale as output to a network?
  but maybe it's not so important for other approaches, so it's fine to skip it?
- train count prediction classifier

dumb baseline:
- predict on small patches
- determine image scale (maybe skip?)
- estimate count based on the number of positive patches (or something like this)

"proper" pipeline ideas:
- ???
- detector that works for all classes
- how to separate close entries - should be solved?
- check SOTA detection models again

later:
- check quality of coordinates carefully on all images
- predict one class (pups) based on other classes:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

