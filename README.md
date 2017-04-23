This script demonstrates how Genetic Algorithms can be used to find "optimal" feature subsets for machine learning problems.

## Links
A YouTube video of me briefly reviewing this project can be found [here](https://www.youtube.com/watch?v=COLO7cGP2sA).

## Inspiration
While working with a data set that I had created, I found myself wondering whether certain measurements should be included in features.  I didn't want to leave anything out, but also didn't want to include so many features that it might negatively impact the performance of some machine learning algorithms.  While reading an academic paper, I came across the concept of using Genetic Algorithms to determine optimal feature subsets.  Therefore, I chose to implement an example of this being done.

## The Future
In the future, I may make a class to specifically facilitate the feature selection process.  This will allow the Genetic Algorithm method of feature selection to be more easily applied "out of the box" to machine learning problems.

## Other Notes
To plot a curve over the noisy data, I used Cubic-Spline Interpolation.  This is my first time using this method, and I suspect there are better ways to plot such a curve.  In my limited experience, Cubic-Spline Interpolation can determine curves that have unnecessary "bends" in them from time to time.
