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


