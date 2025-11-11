Cross Validation
================
2025-11-11

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.2     ✔ tibble    3.3.0
    ## ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ## ✔ purrr     1.1.0     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(modelr)
library(mgcv)
```

    ## Loading required package: nlme
    ## 
    ## Attaching package: 'nlme'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse
    ## 
    ## This is mgcv 1.9-3. For overview type 'help("mgcv-package")'.

``` r
library(p8105.datasets)
```

Load the LIDAR

``` r
data("lidar")

lidar_df =
  lidar |> 
  mutate(id = row_number())

lidar_df |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point()
```

![](cross_validation_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

## Create dataframes

``` r
#training dataset
train_df = sample_frac(lidar_df, size = 0.8) |> 
  arrange(id)

#testing dataframe (antijoin: gives you everything in the larger dataset that's NOT in the smaller one)
test_df = anti_join(lidar_df, train_df, by = "id")
```

Look at these

``` r
ggplot(train_df, aes(x = range, y = logratio)) +
  geom_point() +
  geom_point(data = test_df, color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Fit a few models to the `train_df`

``` r
linear_mod = lm(logratio ~ range, data = train_df)
smooth_mod = mgcv::gam(logratio ~ s(range), data = train_df) 
#nonlinear model fitting mgcv:gam, and then s() gives us something smooth

wiggly_mod = mgcv::gam(logratio ~ s(range, k = 30), sp = 10e-6, data = train_df)
#don't do this
```

Look at this (linear)

``` r
train_df |> 
  add_predictions(linear_mod) |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Look at the smooth one:

``` r
train_df |> 
  add_predictions(smooth_mod) |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Wiggly (again…don’t do this. OVERFITTED)

``` r
train_df |> 
  add_predictions(wiggly_mod) |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

Try computing our RMSEs (root mean square error)

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.1189345

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.07600159

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.08642897

``` r
#smaller rmse usually means better fit
```

## Iterate!

``` r
cv_df =
  crossv_mc(lidar_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

Did this work? Yes!

``` r
cv_df |> pull(train) |> nth(8) #gives e.g. 8th training and testing split
```

    ## # A tibble: 176 × 3
    ##    range logratio    id
    ##    <dbl>    <dbl> <int>
    ##  1   390  -0.0504     1
    ##  2   394  -0.0510     4
    ##  3   396  -0.0599     5
    ##  4   397  -0.0284     6
    ##  5   400  -0.0399     8
    ##  6   402  -0.0294     9
    ##  7   406  -0.0604    12
    ##  8   408  -0.0312    13
    ##  9   409  -0.0382    14
    ## 10   411  -0.0756    15
    ## # ℹ 166 more rows

Let’s fit models over and over

``` r
cv_df =
cv_df |> 
  mutate(
    linear_fits = map(train, \(df) lm(logratio ~ range, data = df)), #using anonymous function
    smooth_fits = map(train, \(df) mgcv::gam(logratio ~ s(range), data = df)),
    wiggly_fits = map(train, \(df) mgcv::gam(logratio ~ s(range, k = 50), sp = 10e-8, data = df)),
  ) |> 
  #Equivalently, can create a function, and then use it. But we just have 1 input, so let's use an anonymous function
  mutate(
    rmse_linear = map2_dbl(linear_fits, test, rmse), #map2 allows 2 input lists (applies a function to 2 inputs) and the dbl shows us the value (uncollapses)
    rmse_smooth = map2_dbl(smooth_fits, test, rmse),
    rmse_wiggly = map2_dbl(wiggly_fits, test, rmse)
  )
```

Let’s try to look at this better

``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse)) + 
  geom_violin()
```

![](cross_validation_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

New example:

``` r
growth_df = read_csv("nepalese_children.csv")
```

    ## Rows: 2705 Columns: 5
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (5): age, sex, weight, height, armc
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

Weight v arm_c

``` r
growth_df |> 
  ggplot(aes(x = weight, y = armc)) + 
  geom_point(alpha = .5)
```

![](cross_validation_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

*Let’s show the models we might use:*

``` r
growth_df =
  growth_df |> 
  mutate(
    weight_cp7 = (weight > 7) * (weight - 7) #change point weight 
  )
```

Let’s fit 3 models:

``` r
linear_mod = lm(armc ~ weight, growth_df) #linear model
pwl_mod    = lm(armc ~ weight + weight_cp7, growth_df) #piece-wise linear model
smooth_mod = mgcv::gam(armc ~ s(~weight), data = growth_df) #smooth model
```

Visualizing goodness of fit for the linear model:

``` r
growth_df |> 
  add_predictions(linear_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = pred, color = "red"))
```

![](cross_validation_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

Does a decent job for the bulk of the data, but at the low end of weight
the prediction line is not through center of data. Linear models are
EASY to interpret.

Visualizing piece-wise linear:

``` r
growth_df |> 
  add_predictions(pwl_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = pred, color = "red"))
```

![](cross_validation_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

Visualizing smooth fit:

``` r
growth_df |> 
  add_predictions(smooth_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = pred, color = "red"))
```

![](cross_validation_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

^Looks similar to piece-wise, but smoothed out. Harder to interpret than
linear.

Time to cross-validate and see which model is best!

Now cross validate:

``` r
cv_df =
  crossv_mc(growth_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

``` r
cv_df =
  cv_df |> 
  mutate(
    linear_mod = map(train, \(df) lm(armc ~ weight, data = df)), #fits this linear model across the training dataframe
    pwl_mod = map(train, \(df) lm(armc ~ weight + weight_cp7, data = df)),
    smooth_mod = map(train, \(df) mgcv::gam(armc ~ s(weight), data = df))
  ) |> 
  mutate(
    rmse_linear = map2_dbl(linear_mod, test, rmse), #compute rmse for the linear model
    rmse_pwl = map2_dbl(pwl_mod, test, rmse), 
    rmse_smooth = map2_dbl(smooth_mod, test, rmse)
  )
```

Create my violin plots:

``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse)) + 
  geom_violin()
```

![](cross_validation_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

Linear model has a higher distribution of higher rmse. Smooth and pwl
seem to perform similarly. But tbh, all of them perform relatively
similary, and you could justify why you prefer each of them most. Might
come down to interpretation.
