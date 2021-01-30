# PokemonDTR

  The purpose of this project is to first predict a numerical relationship between any given pokemon's attack points and special attack points. This will be done upon 
the [pokemon dataset](https://www.kaggle.com/abcsds/pokemon) after loading it in as a [pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html), 
by using [Scikit-Learn's decision tree regressor](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#) algorithm. Two decision trees of varying depth will be used 
to predict the Sp. attack points of several pokemon given attack points. These predictions will then be plotted as a linear regression graph using the [pyplot feature of matplotlib](https://matplotlib.org/tutorials/introductory/pyplot.html).
  Secondly, another decision tree regressor algorithm will be performed to try and predict the numerical relationship between a given pokemon's attack points and 
it's usage percentage in the Smogon pokemon competition. After loading in the Smogon dataset as a dataframe using pandas, an [inner merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)
will be performed between the previous Pokemon dataframe and the Smogon dataframe to yield a new dataframe that only includes pokemon present in both dataframes. 
Then, two more decision tree regressors will be run on the new dataframe to make our predictions for several more pokemon. Finally, this linear regression will be plotted 
as well using matplotlib's pyplot.

Having [Anaconda](https://docs.anaconda.com/) installed and using a virtual environment is highly recommended as one's required packages are easily contained and organized.
For the purposes of this project, numpy will only slightly be required. To install numpy, try `pip install numpy`. For pandas, refer to [this](https://anaconda.org/anaconda/pandas).
For Scikit-Learn, [this](https://scikit-learn.org/stable/install.html). And finally, for matplotlib, [this](https://anaconda.org/anaconda/matplotlib). 
