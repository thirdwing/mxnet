# Customized callback function

This vignette gives users a guideline for using and writing callback functions,
which can be very useful in model training. 

## Model training example

Let's begin from a small example. We can build and train a model using the following code:


```r
library(mxnet)
data(BostonHousing, package="mlbench")
train.ind = seq(1, 506, 3)
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
lro <- mx.symbol.LinearRegressionOutput(fc1)
mx.set.seed(0)
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
```

```
## Start training with 1 devices
```

```
## [1] Train-rmse=16.0632823504593
```

```
## [1] Validation-rmse=10.1766441138363
```

```
## [2] Train-rmse=12.2792377684856
```

```
## [2] Validation-rmse=12.4331765607021
```

```
## [3] Train-rmse=11.1984627645076
```

```
## [3] Validation-rmse=10.3303037342971
```

```
## [4] Train-rmse=10.2645234131736
```

```
## [4] Validation-rmse=8.42760430115982
```

```
## [5] Train-rmse=9.49710982900594
```

```
## [5] Validation-rmse=8.44557796259208
```

```
## [6] Train-rmse=9.07733794061977
```

```
## [6] Validation-rmse=8.33225525132395
```

```
## [7] Train-rmse=9.07884450778845
```

```
## [7] Validation-rmse=8.38827848390959
```

```
## [8] Train-rmse=9.10463869411129
```

```
## [8] Validation-rmse=8.37394425101963
```

```
## [9] Train-rmse=9.03977029041008
```

```
## [9] Validation-rmse=8.25927956688155
```

```
## [10] Train-rmse=8.96870653168924
```

```
## [10] Validation-rmse=8.19509240193974
```

Besides, we provide two optional parameters, `batch.end.callback` and `epoch.end.callback`, which can provide great flexibility in model training.

## How to use callback functions


Two callback functions are provided in this package:

- `mx.callback.save.checkpoint` is used to save checkpoint to files each period iteration.


```r
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.save.checkpoint("boston"))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
```

```
## Start training with 1 devices
```

```
## [1] Train-rmse=19.1621413009259
```

```
## [1] Validation-rmse=20.7215182000477
```

```
## Model checkpoint saved to boston-0001.params
```

```
## [2] Train-rmse=13.5127401548348
```

```
## [2] Validation-rmse=14.1822111721827
```

```
## Model checkpoint saved to boston-0002.params
```

```
## [3] Train-rmse=10.4242998906947
```

```
## [3] Validation-rmse=10.5289625063863
```

```
## Model checkpoint saved to boston-0003.params
```

```
## [4] Train-rmse=9.44009379756303
```

```
## [4] Validation-rmse=10.6310865290327
```

```
## Model checkpoint saved to boston-0004.params
```

```
## [5] Train-rmse=9.3496045752297
```

```
## [5] Validation-rmse=12.4697643293583
```

```
## Model checkpoint saved to boston-0005.params
```

```
## [6] Train-rmse=9.99471463846757
```

```
## [6] Validation-rmse=13.1964717774499
```

```
## Model checkpoint saved to boston-0006.params
```

```
## [7] Train-rmse=9.89798460219192
```

```
## [7] Validation-rmse=12.4948264839432
```

```
## Model checkpoint saved to boston-0007.params
```

```
## [8] Train-rmse=9.53670447278993
```

```
## [8] Validation-rmse=11.9904796568582
```

```
## Model checkpoint saved to boston-0008.params
```

```
## [9] Train-rmse=9.45045703299495
```

```
## [9] Validation-rmse=11.9883154143176
```

```
## Model checkpoint saved to boston-0009.params
```

```
## [10] Train-rmse=9.4594553423549
```

```
## [10] Validation-rmse=12.0914660885518
```

```
## Model checkpoint saved to boston-0010.params
```

```r
list.files(pattern = "^boston")
```

```
##  [1] "boston-0001.params" "boston-0002.params" "boston-0003.params"
##  [4] "boston-0004.params" "boston-0005.params" "boston-0006.params"
##  [7] "boston-0007.params" "boston-0008.params" "boston-0009.params"
## [10] "boston-0010.params" "boston-symbol.json"
```


- `mx.callback.log.train.metric` is used to log training metric each period.
You can use it either as a `batch.end.callback` or a `epoch.end.callback`.


```r
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  batch.end.callback = mx.callback.log.train.metric(5))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
```

```
## Start training with 1 devices
```

```
## Batch [5] Train-rmse=17.651455519829
```

```
## [1] Train-rmse=15.2879605632966
```

```
## [1] Validation-rmse=12.3332066126726
```

```
## Batch [5] Train-rmse=11.939392368788
```

```
## [2] Train-rmse=11.4382239877461
```

```
## [2] Validation-rmse=9.91176525672676
```

```
## Batch [5] Train-rmse=9.38533484167536
```

```
## [3] Train-rmse=9.59719820811031
```

```
## [3] Validation-rmse=9.06276659691733
```

```
## Batch [5] Train-rmse=9.92382946505717
```

```
## [4] Train-rmse=9.75901184915993
```

```
## [4] Validation-rmse=8.49146461970735
```

```
## Batch [5] Train-rmse=9.48338725303142
```

```
## [5] Train-rmse=9.8139315217265
```

```
## [5] Validation-rmse=8.91469462380691
```

```
## Batch [5] Train-rmse=9.43291748718483
```

```
## [6] Train-rmse=9.78568504311423
```

```
## [6] Validation-rmse=9.33040824121574
```

```
## Batch [5] Train-rmse=9.47870400234534
```

```
## [7] Train-rmse=9.55787759894597
```

```
## [7] Validation-rmse=8.8841129281817
```

```
## Batch [5] Train-rmse=9.26720596923921
```

```
## [8] Train-rmse=9.42726088988245
```

```
## [8] Validation-rmse=8.68199749914679
```

```
## Batch [5] Train-rmse=9.16960003361489
```

```
## [9] Train-rmse=9.39504584159076
```

```
## [9] Validation-rmse=8.6347783131459
```

```
## Batch [5] Train-rmse=9.16627873677832
```

```
## [10] Train-rmse=9.36823255103782
```

```
## [10] Validation-rmse=8.57517563255639
```

You can also save the training and evaluation errors for later usage by passing a reference class.


```r
logger <- mx.metric.logger$new()
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
```

```
## Start training with 1 devices
```

```
## [1] Train-rmse=19.1083226599321
```

```
## [1] Validation-rmse=12.715069259381
```

```
## [2] Train-rmse=15.7684373433091
```

```
## [2] Validation-rmse=14.8105299918579
```

```
## [3] Train-rmse=13.53147030123
```

```
## [3] Validation-rmse=15.8403638176235
```

```
## [4] Train-rmse=11.3860504903943
```

```
## [4] Validation-rmse=10.8987325979334
```

```
## [5] Train-rmse=9.55547666490208
```

```
## [5] Validation-rmse=9.3497062675305
```

```
## [6] Train-rmse=9.35132443775407
```

```
## [6] Validation-rmse=9.36308698638248
```

```
## [7] Train-rmse=9.67815291555931
```

```
## [7] Validation-rmse=10.0496001959045
```

```
## [8] Train-rmse=9.79531658980212
```

```
## [8] Validation-rmse=10.3969095345197
```

```
## [9] Train-rmse=9.66005887584128
```

```
## [9] Validation-rmse=10.0821733668574
```

```
## [10] Train-rmse=9.50816948133904
```

```
## [10] Validation-rmse=9.77996492363972
```

```r
head(logger$train)
```

```
## [1] 19.108323 15.768437 13.531470 11.386050  9.555477  9.351324
```

```r
head(logger$eval)
```

```
## [1] 12.715069 14.810530 15.840364 10.898733  9.349706  9.363087
```

## How to write your own callback functions


You can find the source code for two callback functions from [here](https://github.com/dmlc/mxnet/blob/master/R-package/R/callback.R) and they can be used as your template:

Basically, all callback functions follow the structure below:


```r
mx.callback.fun <- function() {
  function(iteration, nbatch, env, verbose) {
  }
}
```

The `mx.callback.save.checkpoint` function below is stateless. It just get the model from environment and save it.


```r
mx.callback.save.checkpoint <- function(prefix, period=1) {
  function(iteration, nbatch, env, verbose=TRUE) {
    if (iteration %% period == 0) {
      mx.model.save(env$model, prefix, iteration)
      if(verbose) message(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
    return(TRUE)
  }
}
```

The `mx.callback.log.train.metric` is a little more complex. It holds a reference class and update it during the training process.


```r
mx.callback.log.train.metric <- function(period, logger=NULL) {
  function(iteration, nbatch, env, verbose=TRUE) {
    if (nbatch %% period == 0 && !is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (nbatch != 0 & verbose)
        message(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value))
      if (!is.null(logger)) {
        if (class(logger) != "mx.metric.logger") {
          stop("Invalid mx.metric.logger.")
        }
        logger$train <- c(logger$train, result$value)
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          if (nbatch != 0 & verbose)
            message(paste0("Batch [", nbatch, "] Validation-", result$name, "=", result$value))
          logger$eval <- c(logger$eval, result$value)
        }
      }
    }
    return(TRUE)
  }
}
```

Now you might be curious why both callback functions `return(TRUE)`.
Can we `return(FALSE)`?

Yes! You can stop the training early by `return(FALSE)`. See the examples below.


```r
mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env, verbose) {
    if (!is.null(env$metric)) {
      if (!is.null(eval.metric)) {
        result <- env$metric$get(env$eval.metric)
        if (result$value < eval.metric) {
          return(FALSE)
        }
      }
    }
    return(TRUE)
  }
}
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.early.stop(10))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
```

```
## Start training with 1 devices
```

```
## [1] Train-rmse=18.5897977284261
```

```
## [1] Validation-rmse=13.5555206982882
```

```
## [2] Train-rmse=12.5867558902911
```

```
## [2] Validation-rmse=9.76304983057768
```

You can see once the validation metric goes below the threshold we set, the training process will stop early.


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
