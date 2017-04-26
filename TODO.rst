- why is cnn_baseline so poor:
  - shallow water looks poor due to few negatives
  - model is very sensitive to precise position - FCN seems to miss a lot
  - neg-5 model might need more training

misc:
- multithreaded dataloader

dumb baseline:
- predict on small patches
- determine image scale (maybe skip?)
- estimate count based on the number of positive patches (or something like this)

UNet:
- train log loss longer, try giving less weight to no lion class
- change loss:
  - add dice, maybe modify it (pos/neg weights, square size)
- separate models for different classes?

"proper" pipeline ideas:
- ???
- detector that works for all classes
- how to separate close entries - should be solved?
- check SOTA detection models again

later:
- check quality of coordinates carefully on all images
- predict one class (pups) based on other classes:
  https://www.kaggle.com/andraszsom/noaa-fisheries-steller-sea-lion-population-count/predict-the-number-of-pups

