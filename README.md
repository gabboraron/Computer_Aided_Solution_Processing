# Computer Aided Solution Processing

- [list of courses about deep learning](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md)
- [Andrew Ng YouTube Lecture Collection | Machine Learning - Stanford](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599)
- [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [free books about deep learning](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md)


**Requirements:**
- test after each lecture 10-15 min
- micro projects

-----

## Topic 1 - Introduction
## Topic 2 - Notation/ Linear Regression / Least Squares Method
### regression types
In supervised learning, if the output `y` is real-valued quantitative, the problem is called regression.

> **Notation**
> - `n` - number of datapoints, the exxamples in training dataset
> - `m` - imput variables
> - `j` - index of columns
> - `i` - index of rows
> - `x`<sub>`j`</sub> - `j`-th input variable
> - `x`<sub>`i`</sub> - `i`-th row vector 
> - `y` output variable

### Regression problem
```
      ____________________
x -> | system under study | -> y
     _____________________
```

When we are studying a behavior of some system, we observe `m` different input variables (features) `x = (x1, x2, ..., xm)` and a quantitative output variable `y`.

`y = h(x) + ε`

`h` represents systematic information that `x` provides about `y`. `ε` is just noise that doesn’t give us any useful information.

- the utput `y` contain noise too!
- we do not know function `h(x)`

We create our function about `x` which gives us a prediction about `y` this will be: `ŷ = f(x)` and can simulate `h`.

### Linear regression
**Linear regression assumes that the relationship between `x` and `y` is liner (simply a sum of the features): `y = a0 + a1x1 + a2x2 + ... + amxm + ε` or ![sum form of equation](https://latex.codecogs.com/gif.latex?y=a_0+\sum_{j=1}^{m}{a_jx_j+\varepsilon})**

Now we just have to find the values of this sum, and this is much easier than find the original function. For linear models, we need to estimate the parameters `a0, a1, ..., am` such that it predicts `y` with the least error. 

#### Simplest case 
one feature => `m=1` => this means just a line: 

`ŷ = a0 + a1x`

- `a0` - determines `y` value when `x = 0`; *ex: when the line crosses the `y` axis - y-intercept*
- `a1` determines the slope of the line

We need to train the model – we need a procedure, that uses the training data to estimate model’s parameters `a0`, `a1` => ***The line should be in minimal distance from all the data points at the same time***

**Criterions:**
- minimize absolute residuals
- minimize residuals with sum of squeres

**How to do it:**
1. try to guess - not so useful
2. [search iteratively](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931) - so much time...
3. [least squeres method](https://setosa.io/ev/ordinary-least-squares-regression/)

model: `ŷ = a0 + a1x` - from `x`calculate `y`, sust as simple with the criteria: ![S=min\left(\sum_{i=1}^{n}\left(l_i\right)^2\right)](https://latex.codecogs.com/gif.latex?S=min\left(\sum_{i=1}^{n}\left(l_i\right)^2\right)) and `li = yi – ŷi`

if we replace `ŷ` and `li` then we got: ![S=min\left(\sum_{i=1}^{n}\left(y_i-a_0-a_1x_i\right)^2\right)](https://latex.codecogs.com/gif.latex?S=min\left(\sum_{i=1}^{n}\left(y_i-a_0-a_1x_i\right)^2\right))

To find the minimal of the function we need to derivate it, on `a0` and `a1` too!

- ![\frac{\partialS}{\partiala_0}=-2\sum_{i=1}^{n}{(y_i-a_0-a_1x_i)}](https://latex.codecogs.com/gif.latex?\frac{dS}{da_0}=-2\sum_{i=1}^{n}{(y_i-a_0-a_1x_i)})
- ![\frac{dS}{da_1}=-2\sum_{i=1}^{n}{x_i(y_i-a_0-a_1x_i)}](https://latex.codecogs.com/gif.latex?\frac{dS}{da_1}=-2\sum_{i=1}^{n}{x_i(y_i-a_0-a_1x_i)})

***in practice the differentiation step is skipped, i.e., we can go directly to the system of equations***

we got a linear system of equations:
- ![fst eq: na_0+a_1\sum_{i=1}^{n}x_i=\sum_{i=1}^{n}y_i](https://latex.codecogs.com/gif.latex?na_0+a_1\sum_{i=1}^{n}x_i=\sum_{i=1}^{n}y_i)
- ![snd eq: a_0\sum_{i=1}^{n}{x_i+}a_1\sum_{i=1}^{n}x_i^2=\sum_{i=1}^{n}{x_iy}_i](https://latex.codecogs.com/gif.latex?a_0\sum_{i=1}^{n}{x_i+}a_1\sum_{i=1}^{n}x_i^2=\sum_{i=1}^{n}{x_iy}_i)


training dataset: `x` is `Exam`
```
Quiz   Exam
5.6    5.0
6.5    7.1
6.8    8.4
6.9    7.3
7.0    7.8
7.4    8.1
8.0    7.4
8.3    8.9
8.7    9.0
9.0    10.0
```

put the traing numbers into the equations... OR use Gaussian elimination

The given line will be the best with this criteria! If we change it then this not will be true!

## Topic 3 - Linear regression | Least square method | simple model evaluation
### linear regressions
- `ŷ = a0+a1 x`
- `ŷ = a0 + a1x + a2x^2`
- `ŷ = a0 + a1x + ... + a2x^d`

#### quadratic polynomial (d = 2 )
- equation for parabola: `ŷ = a0 + a1x + a2x2`
- when a2 -> 0 shape of parabola approaches straight line

now if we have a dataset with `x` and `y`; at least with 14  rows (`n=14`); we want to calculate `a0`, `a1` and `a2` from `ŷ = a0 + a1x + a2x^2`. As previously before, now we have to use `li` in criteria `S` and `ŷ` model in criteria. With this we got: !S\ =\ \sum_{i=1}^{n}{(y_i-a_0{-a}_1x_i-a_2x_i^2)}^2]https://latex.codecogs.com/gif.latex?S\%20=\%20\sum_{i=1}^{n}{(y_i-a_0{-a}_1x_i-a_2x_i^2)}^2) we have to minimize with the Residual Sum of Squares method.

Now we can take partial derivatives of `a0`, `a1`, `a2`
- ![\frac{dS}{da_0}=\ -2\sum_{i=1}^{n}\left(y_i-a_0{-a}_1x_i-a_2x_i^2\right)](https://latex.codecogs.com/gif.latex?\frac{dS}{da_0}=\%20-2\sum_{i=1}^{n}\left(y_i-a_0{-a}_1x_i-a_2x_i^2\right))
- ![\frac{dS}{da_1}=\ -2\sum_{i=1}^{n}{x_i(y_i-a_0{-a}_1x_i-a_2x_i^2)}](https://latex.codecogs.com/gif.latex?\frac{dS}{da_1}=\%20-2\sum_{i=1}^{n}{x_i(y_i-a_0{-a}_1x_i-a_2x_i^2)})
- ![\frac{dS}{da_2}=\ -2\sum_{i=1}^{n}{{x_i}^2(y_i-a_0{-a}_1x_i-a_2x_i^2)}](https://latex.codecogs.com/gif.latex?\frac{dS}{da_2}=\%20-2\sum_{i=1}^{n}{{x_i}^2(y_i-a_0{-a}_1x_i-a_2x_i^2)})

take eqal with 0, and let's make a linear system of equations from it. 
- ![na_0+a_1\sum_{i=1}^{n}x_i+a_2\sum_{i=1}^{n}{x_i}^2=\sum_{i=1}^{n}y_i](https://latex.codecogs.com/gif.latex?na_0+a_1\sum_{i=1}^{n}x_i+a_2\sum_{i=1}^{n}{x_i}^2=\sum_{i=1}^{n}y_i)
- ![a_0\sum_{i=1}^{n}x_i+a_1\sum_{i=1}^{n}{x_i}^2+a_2\sum_{i=1}^{n}{x_i}^3=\sum_{i=1}^{n}{x_iy_i}](https://latex.codecogs.com/gif.latex?a_0\sum_{i=1}^{n}x_i+a_1\sum_{i=1}^{n}{x_i}^2+a_2\sum_{i=1}^{n}{x_i}^3=\sum_{i=1}^{n}{x_iy_i})
- ![a_0\sum_{i=1}^{n}{x_i}^2+a_1\sum_{i=1}^{n}{x_i}^3+a_2\sum_{i=1}^{n}{x_i}^4=\sum_{i=1}^{n}{{x_i}^2y_i}](https://latex.codecogs.com/gif.latex?a_0\sum_{i=1}^{n}{x_i}^2+a_1\sum_{i=1}^{n}{x_i}^3+a_2\sum_{i=1}^{n}{x_i}^4=\sum_{i=1}^{n}{{x_i}^2y_i})

Now we got `a0`, `a1`, `a2`. If we get an `x` value, we can calculate the `ŷ`. 

Any features can be written as a sum of features.

***But what if we need something more complex than just a sum of features?***

#### Least Squares Method generalized

#### how good is our model?
> ***always smaller values are better***

Usually get values on `[0,1]` but if the test dataset is different then training data then might be `-1`, so, in that case, don't use that model **!**

more about different models: https://methods.sagepub.com/images/virtual/heteroskedasticity-in-regression/p53-1.jpg

[Mean Squared Error VS R-Squared](https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/)

## Topic 4 - Model evaluation and selection - Resampling methods
> - how good is a model for oour dataset
> - how to choose the best model for the dataset?

![traingin error](https://drek4537l1klr.cloudfront.net/serrano/v-11/Figures/04_04.png)

- straight line: the model is too simple - R^2: 0.6
- when the courve fit exactly on data points: overfitting: R^2: 0.99

*When the number of parameters is equal to the number of data points the curve goes exactly through all the points*

![prediction vs training](https://vitalflux.com/wp-content/uploads/2020/12/overfitting-and-underfitting-wrt-model-error-vs-complexity.png)

prediction error is nknown, just training error can be calculated, but we can estimate prediction error!

error means: SAE, MAE, SSE, MSE, RMSE

**Advantages of simple models:**
- Better predictive performance (except if underfitted)
- A simple model might use fewer features => fewer measurements or other kinds of data gathering
- faster 
- needs less computer memory
- Simpler models are easier to visualize and interpret

> **Idea**
> 
> *We know that when we use a model for prediction, we calculate y for given x which may not be (and usually is not) given in the training dataset. What if we could “simulate” this process? We could use additional evaluation dataset which is not included in the training dataset.*

### Validation set and test set
- dataset is used for model selection the dataset is called validation set
- it is used for final estimation of prediction error it is called test set

**The aims of model selection and final evaluation:**
- **Model selection:** select the best model by evaluating predictive performance of modelscandidates - relative differences between evaluations of different models
- **Final estimation of the true prediction error:** the true prediction error as closely as possible in this way giving information about the expected error of the model

both cases we can use resampling methods: ***evaluate the model on additional data that was not included in the training***  these additional data points would not be included in the training set, not even in model building

this can happen by:
- Additionally generated
- Simply subtracted from the already existing full dataset

#### The three datasets method
- divide the whole dataset into three subsets (or two, if either validation or testing is not required)
- the division is usually 60:20:20 or 70:30

#### Hold-Out method
- Each model-candidate is evaluated using the validation set
- at given `x` we calculate the error between model’s predicted `ŷ` value and the `y` value given in the validation dataset 
- ***In the end, the one “best” model is evaluated in the same as using the test set***
- the order of the data points is randomized!
- **Advantages:**
  -  Easy to implement and use
  -  **Efficiency of computations** – nothing is to be computed more than once 
- **Disadvantages:**
  - **can't use on small datasets!**
    - Considerably reduces the size of the training data
    - We can get “unlucky” data point combinations in any of the set

![MSE](https://d1zx6djv3kb1v7.cloudfront.net/wp-content/media/2019/11/Differences-between-MSE-and-RMSE-1-i2tutorials.jpg)
 
#### Cross-Validation
##### k-fold Cross-Validation
- All examples of the full dataset are randomly reordered
- divided in `k` subsets of equal size
- `k` iteration:
  - each iteration `j`th subset is used as a test set 
  - other `k – 1` subset is used as a training set
- , the model is created `k` times; error is calculated `k` times => final evaluation of the model: calculating the average of the errors

![k-fold MSE](https://miro.medium.com/max/1072/1*nx_V9ByIgD4IuYb5Yczl0Q.png)

##### Leave-One-Out Cross-Validation *(k = n)*
- number of iterations is equal to the number of examples
- test set always includes only one example but overall through all iterations the evaluation is done on all examples

This is a good alternative for very small datasets because you remove only one example from the training set and you evaluate the model as many times as the possible

Advantages:
- All the data is used for calculations of model parameters
- on small dataset better than Hold-out
disadvantages: 
- requires k times more iterations than othe ones, so this is slow!

## Topic 5 - Alternative to the resampling methods General scheme of the whole process
## Topic 6 - Nearest Neighbors Method
### Nearest Neighbors
Using Euclidean distance method to find the distance of two different points: ![\left \| x_q-x_i \right \|=\sum_{j=1}^{m}(x_{qj}-x_{ij})2](https://latex.codecogs.com/gif.latex?\left%20\|%20x_q-x_i%20\right%20\|=\sum_{j=1}^{m}(x_{qj}-x_{ij})2) on ![x_i=(x_{i1},x_{i2} ... x_{im})](https://latex.codecogs.com/gif.latex?x_i=(x_{i1},x_{i2}%20...%20x_{im}))

- ![xq](https://latex.codecogs.com/gif.latex?x_q) is the query point (a vector) - from here we calculate distances to all training examples
- ![xi](https://latex.codecogs.com/gif.latex?x_i)  the `i`th training example - this is for which we calculate the distance
- `j` is the feature
- `m` is the total number of features
- `n` number of training datapoints

calculate the prediction `ŷ` by simple averaging of `y` over the vound `k` nearest examples: ![\hat{y}=\frac{1}{k}\sum_{i=1}^{k}y_i](https://latex.codecogs.com/gif.latex?\hat{y}=\frac{1}{k}\sum_{i=1}^{k}y_i)

***Larger `k` values reduce influence of noise but smoothes-out predictions!***

To improve predictive performance of k-NN, it is recommended to normalize (rescale values to range [0,1]) ![x_{normalized}=\frac{x-x_{min}}{x_{max}-x_{min}}](https://latex.codecogs.com/gif.latex?x_{normalized}=\frac{x-x_{min}}{x_{max}-x_{min}}) or standardize each feature ![x_{standardized}=\frac{x-\bar{x}}{s}](https://latex.codecogs.com/gif.latex?x_{standardized}=\frac{x-\bar{x}}{s}) where the `s` standardization is ![s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2 }](https://latex.codecogs.com/gif.latex?s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2%20})

> **A potential problem**: all of the `k` neighbors are taken into account with the same weight, i.e., they all have the same influence on the result. *only for `k > 1`; if `k = 1` that's the most complex method*

### Weighted nearest neighbors method
![\hat{y}=\frac{\sum_{i=1}^{k}(w_iy_i)}{\sum_{i=1}^{k}w_i} ](https://latex.codecogs.com/gif.latex?\hat{y}=\frac{\sum_{i=1}^{k}(w_iy_i)}{\sum_{i=1}^{k}w_i})

- Without weighting (uniform): ![w_i=1](https://latex.codecogs.com/gif.latex?w_i=1)
- Inverted distance: ![w_i=1/\left \| x_q-x_i \right \|](https://latex.codecogs.com/gif.latex?w_i=1/\left%20\|%20x_q-x_i%20\right%20\|)
- Predictive performance of `k-NN` can be considerably reduced by having too many features: - Feature subset selection might help

#### advantages
- fast learning - all the examples in memory
- very felxible - able to capture complex patterns using very simple principles
- fast update - more exaplest to training data there is no need in training a new model

#### disadvantages
- serious predictive performance issues for highly dimensional data, especially if the number of data points is small <- feature selection
- needs realatively large computational resources - the learning is fast but the prediction is slow
