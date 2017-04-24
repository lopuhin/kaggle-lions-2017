- why is cnn_baseline so poor, esp. on shallow water?
  - bug in prediction generation?
  - bug in how fcn model is built?
  - poor prediction quality?
  - what else?

misc:
- multithreaded dataloader

dumb baseline:
- predict on small patches
- determine image scale (maybe skip?)
- estimate count based on the number of positive patches (or something like this)

UNet:
- predict big squares
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

