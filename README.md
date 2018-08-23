# Volleyball Tracking

## Introduction

The idea was to build a system capable of tracking a volleyball and maybe even estimating its speed, based on a video file. During 
introductory work it became pretty clear, that there have to be a few constraints on the video quality in order to achieve a reasonable accuracy. These are:
 - the video should be in slow motion - the ball then have a well-defined shape, so it can be localized with much higher precision
 - the background of the video should be still - it is then much easier to detect only moving objects, not to mention estimating ball velocity
 - the ball should be not too far away - working with a two-pixel ball cannot give good results
 - the ball should preferably differ in color from the background as much as possible - it simply makes the detection easier, but
  of course it is not a hard constraint
 
## Model

At first I tried to build a Convolutional Neural Network (CNN) model. After a lot of trials and errors I ended with a little YOLO
(You Only Look Once)-like architecture, especially considering the output (location of the ball). It is not in a classic (x, y) form,
but instead it is a plane of 50x50 boxes, in which only one (containing the center of the ball) is set to 1. This idea seems to give
the best results. Cross-validating on my own dataset it resulted in average precision (AP) of about 0.4 and average localization error
of about 12% (of the frame size). Not a disaster, but also not good enough.

Then I made my research and I found that in one paper they used something called particle filter to track the volleyball.
I tried to apply this for my videos and this technique alone wasn't very useful. But when I combined it with my CNN model it resulted
in quite nice (although still far from perfect) predictions (mean and median localization error dropped to 5-6% and 1-2% respectively). 

## Code

The code is still far from optimal concerning prediction speed and memory usage, but it works. I'll be working on
optimalization. But you can already try it for your own. Download the repository and run this code inside:

```python
import vbt
ims, preds = vbt.predict(video_path='sample_video.mp4', frame_nums=list(range(10, 100)))
```

Wait for a few seconds and that's it! Array ims consists of specified frames, and predictions are stored in preds array. These are
relative predictions, so you have to multiply them by video size in order to have it in pixel format.

(I strongly advice not to use more than 100-200 frame_nums at once. It may result in memory error.)

## Example results

For the example I've chosen a video, for which the results seemed representative of the overall predictive power of the model - that
is the position error is around the mean of all tested videos. So it sometimes does better and sometimes worse than here.

![](https://github.com/kamilkk852/vb-tracking/blob/master/sample_predictions.gif)
