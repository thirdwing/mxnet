# Neural Network with MXNet in Five Minutes

This is the first tutorial for new users of the R package `mxnet`. You will learn to construct a neural network to do regression in 5 minutes.

We will show you how to do classification and regression tasks respectively. The data we use comes from the package `mlbench`.

## Classification

First of all, let us load in the data and preprocess it:


```r
require(mlbench)
```

```
## Loading required package: mlbench
```

```r
require(mxnet)
```

```
## Loading required package: mxnet
```

```
## For more documents, please visit http://mxnet.io
```

```r
data(Sonar, package = "mlbench")

Sonar[,61] <- as.numeric(Sonar[,61])-1
train.ind <- c(1:50, 100:150)
train.x <- data.matrix(Sonar[train.ind, 1:60])
train.y <- Sonar[train.ind, 61]
test.x <- data.matrix(Sonar[-train.ind, 1:60])
test.y <- Sonar[-train.ind, 61]
```

Next we are going to use a multi-layer perceptron (MLP) as our classifier.
In `mxnet`, we have a function called `mx.mlp` so that users can build a general multi-layer neural network to do classification (`out_activation="softmax"`) or regression (`out_activation="rmse"`).
Note for the `softmax` activation, the output is zero-indexed not one-indexed. In the data we use:


```r
table(train.y)
```

```
## train.y
##  0  1 
## 51 50
```

```r
table(test.y)
```

```
## test.y
##  0  1 
## 60 47
```

There are several parameters we have to feed to `mx.mlp`:

- Training data and label.
- Number of hidden nodes in each hidden layers.
- Number of nodes in the output layer.
- Type of the activation.
- Type of the output loss.
- The device to train `mx.gpu()` for GPU or `mx.cpu()` for CPU.
- Other parameters for `mx.model.FeedForward.create`.

The following code piece is showing a possible usage of `mx.mlp`:


```r
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                eval.metric=mx.metric.accuracy)
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-accuracy=0.488888888888889
## [2] Train-accuracy=0.514285714285714

............

## [19] Train-accuracy=0.838095238095238
## [20] Train-accuracy=0.838095238095238
```

Note that `mx.set.seed` is the correct function to control the random process in `mxnet`. You can see the accuracy in each round during training. It is also easy to make prediction and evaluate.

To get an idea of what is happening, we can easily view the computation graph from R.


```r
graph.viz(model$symbol)
```

[<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/knitr/graph.computation.png">](https://github.com/dmlc/mxnet)


```r
preds <- predict(model, test.x)
```

```
## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..
```

```r
pred.label <- max.col(t(preds)) - 1
table(pred.label, test.y)
```

```
##           test.y
## pred.label  0  1
##          0 24 14
##          1 36 33
```

Note for multi-class prediction, mxnet outputs `nclass` x `nexamples`, each each row corresponding to probability of that class.

## Regression

Again, let us preprocess the data first.


```r
data(BostonHousing, package="mlbench")

train.ind <- seq(1, 506, 3)
train.x <- data.matrix(BostonHousing[train.ind, -14])
train.y <- BostonHousing[train.ind, 14]
test.x <- data.matrix(BostonHousing[-train.ind, -14])
test.y <- BostonHousing[-train.ind, 14]
```

Although we can use `mx.mlp` again to do regression by changing the `out_activation`, this time we are going to introduce a flexible way to configure neural networks in `mxnet`. The configuration is done by the "Symbol" system in `mxnet`, which takes care of the links among nodes, the activation, dropout ratio, etc. To configure a multi-layer neural network, we can do it in the following way:


```r
# Define the input data
data <- mx.symbol.Variable("data")
# A fully connected hidden layer
# data: input source
# num_hidden: number of neurons in this hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc1)
```

What matters for a regression task is mainly the last function, this enables the new network to optimize for squared loss. We can now train on this simple data set. In this configuration, we dropped the hidden layer so the input layer is directly connected to the output layer.

next we can make prediction with this structure and other parameters with `mx.model.FeedForward.create`:


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=16.0632823504593
## [2] Train-rmse=12.2792377684856

............

## [49] Train-rmse=8.25728092343136
## [50] Train-rmse=8.24580506495004
```

It is also easy to make prediction and evaluate


```r
preds <- predict(model, test.x)
```

```
## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..
```

```r
sqrt(mean((preds-test.y)^2))
```

```
## [1] 7.800502
```

Currently we have four pre-defined metrics "accuracy", "rmse", "mae" and "rmsle". One might wonder how to customize the evaluation metric. `mxnet` provides the interface for users to define their own metric of interests:


```r
demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
  res <- mean(abs(label-pred))
  return(res)
})
```

This is an example for mean absolute error. We can simply plug it in the training function:


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=demo.metric.mae)
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-mae=13.1889536513016
## [2] Train-mae=9.81431971523497

............

## [49] Train-mae=6.4101128961477
## [50] Train-mae=6.4031249385741
```

In the previous example, our target is to predict the last column ("medv") in the dataset.
It is also possible to build a regression model with multiple outputs.
This time we use the last two columns as the targets:


```r
train.x <- data.matrix(BostonHousing[train.ind, -(13:14)])
train.y <- BostonHousing[train.ind, c(13:14)]
test.x <- data.matrix(BostonHousing[-train.ind, -(13:14)])
test.y <- BostonHousing[-train.ind, c(13:14)]
```

and build a similar network symbol:


```r
data <- mx.symbol.Variable("data")
fc2 <- mx.symbol.FullyConnected(data, num_hidden=2)
lro2 <- mx.symbol.LinearRegressionOutput(fc2)
```

We use `mx.io.arrayiter` to build an iter for our training set and train the model using `mx.model.FeedForward.create`:


```r
mx.set.seed(0)
train_iter = mx.io.arrayiter(data = t(train.x), label = t(train.y))

model <- mx.model.FeedForward.create(lro2, X=train_iter,
                                     ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9)
```

```
## Start training with 1 devices
```

After training, we can see that the dimension of the prediction is the same with our target.


```r
preds <- t(predict(model, test.x))
```

```
## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..
```

```r
dim(preds)
```

```
## [1] 337   2
```

```r
dim(test.y)
```

```
## [1] 337   2
```
Congratulations! Now you have learnt the basic for using `mxnet`. Please check the other tutorials for advanced features.


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
