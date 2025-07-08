Decision Tree & Random Forest on Hitters Salary Data
================
Erin Gregoire,
November 2024

This project will consider the Hitters Salary data. Using a Regression
tree and Random Forest to predict salary, the purpose is to interpet
which features players should focus on to increase their salary.

Preprocessing & Exploratory Data Analysis:

``` r
library(ISLR2)
```

    ## Warning: package 'ISLR2' was built under R version 4.5.1

``` r
data(Hitters)
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 4.5.1

    ## randomForest 4.7-1.2

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(rpart)

?Hitters
```

    ## starting httpd help server ...

    ##  done

``` r
str(Hitters)
```

    ## 'data.frame':    322 obs. of  20 variables:
    ##  $ AtBat    : int  293 315 479 496 321 594 185 298 323 401 ...
    ##  $ Hits     : int  66 81 130 141 87 169 37 73 81 92 ...
    ##  $ HmRun    : int  1 7 18 20 10 4 1 0 6 17 ...
    ##  $ Runs     : int  30 24 66 65 39 74 23 24 26 49 ...
    ##  $ RBI      : int  29 38 72 78 42 51 8 24 32 66 ...
    ##  $ Walks    : int  14 39 76 37 30 35 21 7 8 65 ...
    ##  $ Years    : int  1 14 3 11 2 11 2 3 2 13 ...
    ##  $ CAtBat   : int  293 3449 1624 5628 396 4408 214 509 341 5206 ...
    ##  $ CHits    : int  66 835 457 1575 101 1133 42 108 86 1332 ...
    ##  $ CHmRun   : int  1 69 63 225 12 19 1 0 6 253 ...
    ##  $ CRuns    : int  30 321 224 828 48 501 30 41 32 784 ...
    ##  $ CRBI     : int  29 414 266 838 46 336 9 37 34 890 ...
    ##  $ CWalks   : int  14 375 263 354 33 194 24 12 8 866 ...
    ##  $ League   : Factor w/ 2 levels "A","N": 1 2 1 2 2 1 2 1 2 1 ...
    ##  $ Division : Factor w/ 2 levels "E","W": 1 2 2 1 1 2 1 2 2 1 ...
    ##  $ PutOuts  : int  446 632 880 200 805 282 76 121 143 0 ...
    ##  $ Assists  : int  33 43 82 11 40 421 127 283 290 0 ...
    ##  $ Errors   : int  20 10 14 3 4 25 7 9 19 0 ...
    ##  $ Salary   : num  NA 475 480 500 91.5 750 70 100 75 1100 ...
    ##  $ NewLeague: Factor w/ 2 levels "A","N": 1 2 1 2 2 1 1 1 2 1 ...

``` r
which(is.na(Hitters) == TRUE) # quite a lot of entries with missing data
```

    ##  [1] 5797 5812 5815 5819 5827 5829 5833 5835 5836 5838 5839 5841 5845 5849 5854
    ## [16] 5861 5863 5866 5868 5874 5877 5880 5891 5894 5898 5900 5901 5902 5903 5911
    ## [31] 5922 5935 5941 5947 5954 5955 5957 5966 5968 5970 5994 5996 6000 6005 6007
    ## [46] 6022 6025 6032 6043 6047 6050 6051 6067 6080 6089 6095 6099 6102 6113

``` r
hitters <- na.omit(Hitters)

hist(hitters$Salary, xlab = "Salary", main = "Distribution of Salary in the MLB")
```

![](Decision-Tree---Random-Forest-on-Hitters-Salary-Data_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->
Taking a quick look at the Salary to get a feel for the response
variable shows that the distribution is highly skewed.

``` r
hitters$Salary <- log(hitters$Salary)
hist(hitters$Salary, xlab = "Log Transformation of Salary", main = "Distribution of Log Salary in the MLB")
```

![](Decision-Tree---Random-Forest-on-Hitters-Salary-Data_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->
A log transformation of the salary field was completed so that the
response variable begins to resemble a normal bell curve distribution.

``` r
set.seed(999)
indis <- sample(1:nrow(hitters), 2/3*round(nrow(hitters)), replace = FALSE)

train <- hitters[indis, ] #175
test <- hitters[-indis, ] #88
```

Fitting a Decision Tree:

``` r
model.control <- rpart.control(minbucket = 2, minsplit = 4, xval = 10, cp = 0)
fit.hitters <- rpart(Salary ~ ., data = train, control = model.control)
plot(fit.hitters, branch = .3, compress = T, uniform = TRUE, margin = .1, main = "Unpruned Tree on Hitters Data")
text(fit.hitters, cex = .5)
```

![](Decision-Tree---Random-Forest-on-Hitters-Salary-Data_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->
This graph shows the original unpruned regression tree to predict salary
on the Hitters Data. It is very messy and difficult to interpret.

``` r
fit.hitters$cptable
```

    ##              CP nsplit  rel error    xerror       xstd
    ## 1  5.453965e-01      0 1.00000000 1.0098049 0.07900462
    ## 2  6.790173e-02      1 0.45460353 0.5097998 0.07601101
    ## 3  5.614607e-02      3 0.31880007 0.4801627 0.07912660
    ## 4  3.383982e-02      4 0.26265400 0.4234388 0.08448950
    ## 5  1.513559e-02      5 0.22881418 0.3324986 0.05063178
    ## 6  1.512499e-02      6 0.21367859 0.3804069 0.05802570
    ## 7  1.431769e-02      7 0.19855360 0.3799743 0.05803500
    ## 8  1.401862e-02      8 0.18423591 0.3786255 0.05806101
    ## 9  9.348291e-03      9 0.17021729 0.3967662 0.06065486
    ## 10 8.780653e-03     10 0.16086899 0.4110475 0.06116429
    ## 11 8.722772e-03     11 0.15208834 0.4110475 0.06116429
    ## 12 8.238586e-03     14 0.12592002 0.4115198 0.06113401
    ## 13 7.362451e-03     15 0.11768144 0.4083104 0.06099845
    ## 14 6.096482e-03     16 0.11031899 0.4008200 0.06089206
    ## 15 5.625016e-03     17 0.10422251 0.3968790 0.05957657
    ## 16 4.598593e-03     18 0.09859749 0.3909229 0.06035736
    ## 17 4.404868e-03     19 0.09399890 0.4002820 0.06264527
    ## 18 4.108364e-03     20 0.08959403 0.4053906 0.06302782
    ## 19 3.946505e-03     21 0.08548566 0.4176430 0.06559812
    ## 20 3.913018e-03     22 0.08153916 0.4166279 0.06568085
    ## 21 3.898386e-03     23 0.07762614 0.4073486 0.06245492
    ## 22 3.522576e-03     24 0.07372776 0.4077270 0.06240243
    ## 23 3.282312e-03     25 0.07020518 0.4192770 0.06324872
    ## 24 3.268903e-03     26 0.06692287 0.4264192 0.06547553
    ## 25 2.934040e-03     27 0.06365396 0.4422757 0.06667022
    ## 26 2.522477e-03     28 0.06071992 0.4395087 0.06619546
    ## 27 2.513730e-03     29 0.05819745 0.4419421 0.06629487
    ## 28 2.447900e-03     30 0.05568372 0.4413224 0.06629253
    ## 29 1.917632e-03     31 0.05323582 0.4417530 0.06632779
    ## 30 1.911061e-03     32 0.05131819 0.4535688 0.06689349
    ## 31 1.830046e-03     33 0.04940712 0.4534148 0.06689917
    ## 32 1.777293e-03     34 0.04757708 0.4544282 0.06678250
    ## 33 1.498630e-03     35 0.04579979 0.4536978 0.06566286
    ## 34 1.306210e-03     36 0.04430116 0.4601044 0.06633897
    ## 35 1.240691e-03     37 0.04299495 0.4567196 0.06619522
    ## 36 1.228624e-03     38 0.04175425 0.4571205 0.06618059
    ## 37 1.183889e-03     39 0.04052563 0.4556346 0.06618467
    ## 38 1.171068e-03     40 0.03934174 0.4561522 0.06619811
    ## 39 9.773354e-04     41 0.03817067 0.4563850 0.06621260
    ## 40 9.448700e-04     42 0.03719334 0.4555893 0.06623110
    ## 41 8.314418e-04     43 0.03624847 0.4567817 0.06618760
    ## 42 8.235746e-04     44 0.03541703 0.4568287 0.06614718
    ## 43 8.111205e-04     45 0.03459345 0.4554425 0.06612954
    ## 44 7.434132e-04     46 0.03378233 0.4559953 0.06611075
    ## 45 6.826332e-04     47 0.03303892 0.4551614 0.06612911
    ## 46 5.987917e-04     48 0.03235629 0.4596626 0.06753585
    ## 47 5.421646e-04     49 0.03175749 0.4596661 0.06754132
    ## 48 5.060708e-04     50 0.03121533 0.4600489 0.06755072
    ## 49 4.231331e-04     51 0.03070926 0.4608361 0.06754565
    ## 50 4.198459e-04     52 0.03028613 0.4607326 0.06753059
    ## 51 4.164856e-04     53 0.02986628 0.4620137 0.06754746
    ## 52 3.627550e-04     54 0.02944979 0.4618757 0.06755103
    ## 53 3.128058e-04     55 0.02908704 0.4615924 0.06747044
    ## 54 3.089623e-04     56 0.02877423 0.4646103 0.06812605
    ## 55 2.528563e-04     57 0.02846527 0.4617411 0.06734778
    ## 56 2.332760e-04     58 0.02821241 0.4619981 0.06734207
    ## 57 2.271035e-04     59 0.02797914 0.4619981 0.06734207
    ## 58 1.782249e-04     60 0.02775203 0.4619001 0.06733667
    ## 59 1.538522e-04     61 0.02757381 0.4603592 0.06672978
    ## 60 1.515828e-04     62 0.02741996 0.4602151 0.06673430
    ## 61 1.498613e-04     63 0.02726837 0.4602151 0.06673430
    ## 62 1.195435e-04     64 0.02711851 0.4607309 0.06674194
    ## 63 1.044164e-04     65 0.02699897 0.4611722 0.06672714
    ## 64 9.921089e-05     66 0.02689455 0.4611925 0.06672251
    ## 65 9.400136e-05     67 0.02679534 0.4611925 0.06672251
    ## 66 8.402708e-05     68 0.02670134 0.4612496 0.06668163
    ## 67 8.172640e-05     69 0.02661731 0.4612496 0.06668163
    ## 68 7.464838e-05     70 0.02653559 0.4614410 0.06667531
    ## 69 6.248499e-05     71 0.02646094 0.4609030 0.06665322
    ## 70 5.769698e-05     72 0.02639845 0.4600333 0.06664021
    ## 71 3.797545e-05     73 0.02634076 0.4600618 0.06664371
    ## 72 3.400909e-05     74 0.02630278 0.4600618 0.06664371
    ## 73 1.734444e-05     75 0.02626877 0.4603623 0.06663400
    ## 74 2.324689e-06     76 0.02625143 0.4611129 0.06697803
    ## 75 0.000000e+00     77 0.02624910 0.4611073 0.06697814

``` r
min_cp <- which.min(fit.hitters$cptable[,4])
min_cp
```

    ## 5 
    ## 5

``` r
pruned.hitters <- prune(fit.hitters, cp = fit.hitters$cptable[min_cp,1])
plot(pruned.hitters, compress = T, uniform = TRUE, margin = .1, main = "Pruned Tree on Hitters Data")
text(pruned.hitters, cex = .5)
```

![](Decision-Tree---Random-Forest-on-Hitters-Salary-Data_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->
This plot shows the tree after it has been pruned. It is much cleaner
and easier to interpret with less variables.

``` r
pred_train <- predict(pruned.hitters, newdata = train)
train_mse_tree <- mean((pred_train - train$Salary)^2)
train_mse_tree  
```

    ## [1] 0.1731488

``` r
pred_test <- predict(pruned.hitters, newdata = test)
test_mse_tree <- mean((pred_test - test$Salary)^2)
test_mse_tree
```

    ## [1] 0.1950421

Implementing Random Forest:

``` r
rf.hitters <- randomForest(Salary ~ ., data = train, n.tree = 1000)

pred_rf_train <- predict(rf.hitters, newdata = train)
train_mse_rf <- mean((pred_rf_train - train$Salary)^2)
train_mse_rf
```

    ## [1] 0.03923269

``` r
pred_rf_test <- predict(rf.hitters, newdata = test)
test_mse_rf <- mean((pred_rf_test - test$Salary)^2)
test_mse_rf
```

    ## [1] 0.1228175

``` r
importance(rf.hitters)
```

    ##           IncNodePurity
    ## AtBat         4.5277282
    ## Hits          4.6912035
    ## HmRun         2.4435039
    ## Runs          4.1085640
    ## RBI           4.0180964
    ## Walks         5.1927125
    ## Years         3.7227436
    ## CAtBat       20.7507403
    ## CHits        23.8396029
    ## CHmRun        5.2759192
    ## CRuns        19.2131038
    ## CRBI         13.4875842
    ## CWalks       12.1969576
    ## League        0.1879568
    ## Division      0.2571327
    ## PutOuts       2.3735918
    ## Assists       1.2749524
    ## Errors        1.0601139
    ## NewLeague     0.2270678

``` r
varImpPlot(rf.hitters, main = "Variable Importance to Predict Hitters' Salary")
```

![](Decision-Tree---Random-Forest-on-Hitters-Salary-Data_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->
Based on the variable importance table and plot, there are three main
groupings that variables fall into. The first chunk that categorizes the
most important variables are CHits, CAtBat, CRuns, CRBI and CWalks. All
five of these variables are related to how well a baseball player
performed in their overall career. The second grouping of variables are
the next most important including CHmRun, Walks, Hits, AtBat, Runs, RBI,
and Years. The majority of these are again related to a player’s
performance, but looking only at one year of their play history rather
than their career as a whole. The last group of variables are what are
deemed the least important although they may still provide relevant data
for assisting predictions. These variables include HmRun, PutOuts,
Assists, Errors, Division, NewLeague, and League. Of these, the last
three variables especially seem to not be important to predicting
salary. This means that where a player plays (league, division) are the
least characterizing features that relate to how much the player will
earn.

If a player wants to increase their statistics to work towards a higher
salary, they should focus on their performance during each season. High
performance during a single season is categorized by the second most
important grouping on the variable importance plot. Each season’s
performance forms the player’s overall performance, which contain the
most important factors at increasing salary.

Evaluating and Interpreting Results:

``` r
Model = c('CART', 'Random Forest')
Train_MSE = c(round(train_mse_tree, 4), round(train_mse_rf, 4))
Test_MSE = c(round(test_mse_tree, 4), round(test_mse_rf, 4))
Performance_Table <- data.frame(Model, Train_MSE, Test_MSE)
Performance_Table
```

    ##           Model Train_MSE Test_MSE
    ## 1          CART    0.1731   0.1950
    ## 2 Random Forest    0.0392   0.1228

Overall, random forest did a better job at predicting a baseball
player’s salary. This is shown by approximately 13 percent improvement
on the training data and 7 percent improvement on the test data. The
final pruned tree found that the most important variables are CRuns,
Runs, AtBat, CAtBat, and CRBI. Random Forest also found these variables
to be of importance, although they were characterized by different
levels of importance that in the decision tree. It also appears the
Random Forest found more variables to be important than did CART, which
may be one of the reasons why Random Forest drastically outperformed
CART. One thing that stood out was the incredibly low MSE for Random
Forest on the training data. This then led to an incredibly high jump in
performance when Random Forest was conducted on the test data. However,
both outperformed the CART model regardless.
