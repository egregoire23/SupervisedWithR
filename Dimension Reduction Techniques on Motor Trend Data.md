Dimension Reduction Techniques on Motor Trend Data
================
Erin Gregoire,
October 2024

The purpose of this project is to gain experience working with four
popular dimensionality reduction techniques, forward and backward subset
selection, lasso, and ridge by implementing them on the Motor Trend
automobile data and evaluating their performance.

Preprocessing & Exloratory Data Analysis:

``` r
data(mtcars)
library(leaps)
```

    ## Warning: package 'leaps' was built under R version 4.5.1

``` r
library(glmnet)
```

    ## Warning: package 'glmnet' was built under R version 4.5.1

    ## Loading required package: Matrix

    ## Loaded glmnet 4.1-9

``` r
?mtcars
```

    ## starting httpd help server ...

    ##  done

``` r
str(mtcars)
```

    ## 'data.frame':    32 obs. of  11 variables:
    ##  $ mpg : num  21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 ...
    ##  $ cyl : num  6 6 4 6 8 6 8 4 4 6 ...
    ##  $ disp: num  160 160 108 258 360 ...
    ##  $ hp  : num  110 110 93 110 175 105 245 62 95 123 ...
    ##  $ drat: num  3.9 3.9 3.85 3.08 3.15 2.76 3.21 3.69 3.92 3.92 ...
    ##  $ wt  : num  2.62 2.88 2.32 3.21 3.44 ...
    ##  $ qsec: num  16.5 17 18.6 19.4 17 ...
    ##  $ vs  : num  0 0 1 1 0 1 0 1 1 1 ...
    ##  $ am  : num  1 1 1 0 0 0 0 0 0 0 ...
    ##  $ gear: num  4 4 4 3 3 3 3 4 4 4 ...
    ##  $ carb: num  4 4 1 1 2 1 4 2 2 4 ...

``` r
head(mtcars)
```

    ##                    mpg cyl disp  hp drat    wt  qsec vs am gear carb
    ## Mazda RX4         21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
    ## Mazda RX4 Wag     21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
    ## Datsun 710        22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
    ## Hornet 4 Drive    21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
    ## Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
    ## Valiant           18.1   6  225 105 2.76 3.460 20.22  1  0    3    1

``` r
which(is.na(mtcars) == TRUE)
```

    ## integer(0)

``` r
mtcars$vs <- as.factor(mtcars$vs)
mtcars$am <- as.factor(mtcars$am)
```

The first step I took to pre-process the data is to check for missing
values. Fortunately, this dataset does not have any missing data. Next,
I recoded the two binary variables, Engine (vs) and Transmission (am) as
factors. The next step is to visualize the data and get a feel for which
variables may be correlated.

``` r
plot(mtcars)
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

This basic pairs plot shows a breakdown of how each variable relates to
each other. This plot is a useful starting point to get a scope of which
variables should be further explored.

``` r
boxplot(mtcars, main="Boxplot of MTCars Data", xlab="Features", ylab="Range")
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

This plot shows the distribution of each variable. In this mtcars data
set, this boxplot simply shows that besides “disp” and “hp” we are
working with variables that have small distributions.

``` r
hist(mtcars$mpg, main="Distribution of Vehicles' MPG")
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Since we will be predicting the MPG of cars in this data set, it is
helpful to see a general breakdown of how the mpg is distributed across
the data set.

``` r
plot(mtcars$am, mtcars$mpg, xlab = "Transmission", ylab = "MPG", main="Transmission Types by MPG")
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

This plot shows the distribution across Miles Per Gallon when the
transmission is automatic (0) and when the transmission is manual (1).
The range of manual transmissions appears to yield a higher MPG than
automatic transmissions.

``` r
plot(mtcars$vs, mtcars$mpg, xlab = "Engine", ylab = "MPG", main="Engine Types by MPG")
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

This plot shows the distributions of engines on MPG. 0 stands for
V-shaped engines and 1 stands for straight engines. From this boxplot,
it appears that straight engines have a higher average MPG than v-shaped
engines.

Implementing Forward Subset Selection:

``` r
fwd.fit <- regsubsets(mpg ~ ., data = mtcars, nbest = 1, nvmax = 10, method = "forward")
summary(fwd.fit)
```

    ## Subset selection object
    ## Call: regsubsets.formula(mpg ~ ., data = mtcars, nbest = 1, nvmax = 10, 
    ##     method = "forward")
    ## 10 Variables  (and intercept)
    ##      Forced in Forced out
    ## cyl      FALSE      FALSE
    ## disp     FALSE      FALSE
    ## hp       FALSE      FALSE
    ## drat     FALSE      FALSE
    ## wt       FALSE      FALSE
    ## qsec     FALSE      FALSE
    ## vs1      FALSE      FALSE
    ## am1      FALSE      FALSE
    ## gear     FALSE      FALSE
    ## carb     FALSE      FALSE
    ## 1 subsets of each size up to 10
    ## Selection Algorithm: forward
    ##           cyl disp hp  drat wt  qsec vs1 am1 gear carb
    ## 1  ( 1 )  " " " "  " " " "  "*" " "  " " " " " "  " " 
    ## 2  ( 1 )  "*" " "  " " " "  "*" " "  " " " " " "  " " 
    ## 3  ( 1 )  "*" " "  "*" " "  "*" " "  " " " " " "  " " 
    ## 4  ( 1 )  "*" " "  "*" " "  "*" " "  " " "*" " "  " " 
    ## 5  ( 1 )  "*" " "  "*" " "  "*" "*"  " " "*" " "  " " 
    ## 6  ( 1 )  "*" "*"  "*" " "  "*" "*"  " " "*" " "  " " 
    ## 7  ( 1 )  "*" "*"  "*" "*"  "*" "*"  " " "*" " "  " " 
    ## 8  ( 1 )  "*" "*"  "*" "*"  "*" "*"  " " "*" "*"  " " 
    ## 9  ( 1 )  "*" "*"  "*" "*"  "*" "*"  " " "*" "*"  "*" 
    ## 10  ( 1 ) "*" "*"  "*" "*"  "*" "*"  "*" "*" "*"  "*"

``` r
fwd <- summary(fwd.fit)
coef(fwd.fit, 10)
```

    ## (Intercept)         cyl        disp          hp        drat          wt 
    ## 12.30337416 -0.11144048  0.01333524 -0.02148212  0.78711097 -3.71530393 
    ##        qsec         vs1         am1        gear        carb 
    ##  0.82104075  0.31776281  2.52022689  0.65541302 -0.19941925

``` r
par(mfrow = c(2, 2))
plot(fwd$bic, xlab = 'No. of variables', ylab = 'BIC', type = "b", main = "BIC on Forward Subset Selection")
plot(fwd$cp, xlab = 'No. of variables', ylab = 'Cp', type = "b", main = "Cp on Forward Subset Selection")
plot(fwd$rss, xlab = 'No. of variables', ylab = 'RSS', type = "b", main = "RSS on Forward Subset Selection")
plot(fwd$adjr2, xlab = 'No. of variables', ylab = 'Adjusted Rsq', type = "b", main = "Adjusted RSq on Forward Subset Selection")
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The forward subset selection model shows that the most important
predictor variable is weight, followed by number of cylinders and gross
horsepower. Forward subset selection found that a straight engine (vs1)
and number of carburetors had the least impact on predicting miles per
gallon. Based on the graphs of performance indicators, all indicators
point to a two or three variable model.

Implementing Backward Subset Selection:

``` r
bwd.fit <- regsubsets(mpg ~ ., data = mtcars, nbest = 1, nvmax = 10, method = "backward")
summary(bwd.fit)
```

    ## Subset selection object
    ## Call: regsubsets.formula(mpg ~ ., data = mtcars, nbest = 1, nvmax = 10, 
    ##     method = "backward")
    ## 10 Variables  (and intercept)
    ##      Forced in Forced out
    ## cyl      FALSE      FALSE
    ## disp     FALSE      FALSE
    ## hp       FALSE      FALSE
    ## drat     FALSE      FALSE
    ## wt       FALSE      FALSE
    ## qsec     FALSE      FALSE
    ## vs1      FALSE      FALSE
    ## am1      FALSE      FALSE
    ## gear     FALSE      FALSE
    ## carb     FALSE      FALSE
    ## 1 subsets of each size up to 10
    ## Selection Algorithm: backward
    ##           cyl disp hp  drat wt  qsec vs1 am1 gear carb
    ## 1  ( 1 )  " " " "  " " " "  "*" " "  " " " " " "  " " 
    ## 2  ( 1 )  " " " "  " " " "  "*" "*"  " " " " " "  " " 
    ## 3  ( 1 )  " " " "  " " " "  "*" "*"  " " "*" " "  " " 
    ## 4  ( 1 )  " " " "  "*" " "  "*" "*"  " " "*" " "  " " 
    ## 5  ( 1 )  " " "*"  "*" " "  "*" "*"  " " "*" " "  " " 
    ## 6  ( 1 )  " " "*"  "*" "*"  "*" "*"  " " "*" " "  " " 
    ## 7  ( 1 )  " " "*"  "*" "*"  "*" "*"  " " "*" "*"  " " 
    ## 8  ( 1 )  " " "*"  "*" "*"  "*" "*"  " " "*" "*"  "*" 
    ## 9  ( 1 )  " " "*"  "*" "*"  "*" "*"  "*" "*" "*"  "*" 
    ## 10  ( 1 ) "*" "*"  "*" "*"  "*" "*"  "*" "*" "*"  "*"

``` r
bwd <- summary(bwd.fit)
coef(bwd.fit, 10)
```

    ## (Intercept)         cyl        disp          hp        drat          wt 
    ## 12.30337416 -0.11144048  0.01333524 -0.02148212  0.78711097 -3.71530393 
    ##        qsec         vs1         am1        gear        carb 
    ##  0.82104075  0.31776281  2.52022689  0.65541302 -0.19941925

``` r
par(mfrow = c(2, 2))
plot(bwd$bic, xlab = 'No. of variables', ylab = 'BIC', type = "b", main = "BIC on Backward Subset Selection")
plot(bwd$cp, xlab = 'No. of variables', ylab = 'Cp', type = "b", main = "Cp on Backward Subset Selection")
plot(bwd$rss, xlab = 'No. of variables', ylab = 'RSS', type = "b", main = "RSS on Backward Subset Selection")
plot(bwd$adjr2, xlab = 'No. of variables', ylab = 'Adjusted RSq', type = "b", main = "Adjusted RSq on Backward Subset Selection")
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

Backwards subset selection shows that the most important predictor
variables are weight, quarter mile time, having a manual transmission
(am1), and horsepower. The predictor variable that has the least impact
when predicting mpg is the number of cylinders, followed by having a
straight engine and the number of carburetors. The performance
indicators shows that a three or four variable model would demonstrate
parsimony.

Fitting a Ridge Model:

``` r
ridge <- glmnet(mtcars[ ,2:11], mtcars$mpg, alpha = 0)
summary(ridge)
```

    ##           Length Class     Mode   
    ## a0         100   -none-    numeric
    ## beta      1000   dgCMatrix S4     
    ## df         100   -none-    numeric
    ## dim          2   -none-    numeric
    ## lambda     100   -none-    numeric
    ## dev.ratio  100   -none-    numeric
    ## nulldev      1   -none-    numeric
    ## npasses      1   -none-    numeric
    ## jerr         1   -none-    numeric
    ## offset       1   -none-    logical
    ## call         4   -none-    call   
    ## nobs         1   -none-    numeric

``` r
dim(coef(ridge))
```

    ## [1]  11 100

``` r
plot(ridge, xvar = "lambda", main = "Ridge Model on MPG", label = TRUE)
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

Taking a look at the first ridge model which contained 100 different
values for lambda, we see that as lambda increases, the coefficients
shrink towards zero. The coefficients for 2 and 3 (display and
horsepower) remain very stable as lambda increases, meaning that it is
not sensitive to regularization and may hold a consistent effect on MPG.

``` r
ridge.lambda <- min(ridge$lambda)
best.ridge <- predict(ridge, s = ridge.lambda, type = "coefficients")
best.ridge
```

    ## 11 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s0
    ## (Intercept) 16.831454676
    ## cyl         -0.249615804
    ## disp        -0.001941782
    ## hp          -0.013068281
    ## drat         0.981164133
    ## wt          -1.892240792
    ## qsec         0.313425238
    ## vs           0.482628428
    ## am           2.111716247
    ## gear         0.632184273
    ## carb        -0.661753198

Then, a ridge model was fit using the best lambda which is the smallest
with the least variance. This model shows the coefficient for
transmission (am) carries the most significance as a predictor variable,
followed by weight as the second most important predictor variable. The
ridge model shows that the least impactful variable when predicting the
mpg is displacement, followed by horsepower, which is interesting
considering these two variables were shown in the full model graph of
ridge as having the least sensitivity to lambda.

Fitting a LASSO Model:

``` r
lasso <- glmnet(mtcars[ ,2:11], mtcars$mpg, alpha = 1)
summary(lasso)
```

    ##           Length Class     Mode   
    ## a0         79    -none-    numeric
    ## beta      790    dgCMatrix S4     
    ## df         79    -none-    numeric
    ## dim         2    -none-    numeric
    ## lambda     79    -none-    numeric
    ## dev.ratio  79    -none-    numeric
    ## nulldev     1    -none-    numeric
    ## npasses     1    -none-    numeric
    ## jerr        1    -none-    numeric
    ## offset      1    -none-    logical
    ## call        4    -none-    call   
    ## nobs        1    -none-    numeric

``` r
dim(coef(lasso))
```

    ## [1] 11 79

``` r
plot(lasso, xvar = "lambda", main = "LASSO Model on MPG", label = TRUE)
```

![](Dimension-Reduction-Techniques-on-Motor-Trend-Data_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

This graph shows that as lambda has a higher penalty and the predictor’s
coefficients begin to shrink towards zero, the end result is a very
simplified model. It is very distinct how many of the predictors’
coefficients drop off around the -2 to -1 log lambda point. Only 5
(weight) and 1 (cylinders) make it past the majority of coefficient drop
offs.

``` r
lasso.lambda <- min(lasso$lambda)
best.lasso <- predict(lasso, s = lasso.lambda, type = "coefficients")
best.lasso
```

    ## 11 x 1 sparse Matrix of class "dgCMatrix"
    ##                      s0
    ## (Intercept)  9.76652015
    ## cyl         -0.09683688
    ## disp         0.01178995
    ## hp          -0.02048510
    ## drat         0.80052488
    ## wt          -3.58989635
    ## qsec         0.79626195
    ## vs           0.29788827
    ## am           2.50193332
    ## gear         0.65197334
    ## carb        -0.24339282

The coefficients of the LASSO model show that the weight by far is the
most significant predictor variable for predicting MPG, followed in
second by transmission (am). In this model, displacement and gross
horsepower appear to be the least effective in predicting miles per
gallon.

Evaluation & Comparison:

When looking at the original scatterplot from the exploratory analysis,
we expect the most correlation to be within the variables of weight,
drat (rear axle ration), horsepower, displacement, and qsec (quarter
mile time). Between all four models, Forward and Backward subset
selection, Ridge, and LASSO, it is agreed upon the weight is the most
important predictor variable for MPG. Other than this conclusion, the
other predictor variables vary between each model. Forward and Backward
subset selection performed more similarly to each other and LASSO and
ridge performed more similarly to each other. One important variable
that stands out is gross horsepower. Forward and backward subset
selection chose horsepower as one of the most important predictors.
However, Ridge and LASSO both had horsepower as one of the least
important predictors. Also, Ridge and LASSO found transmission to be a
more important predictor than did subset selection.
