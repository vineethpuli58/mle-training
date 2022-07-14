# Median housing value prediction

The housing data can be downloaded from https://github.com/ageron/handson-ml/tree/master/datasets/housing/housing.csv. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
 - Download the files (env.yml, nonstandardcode_modified.py)
 - On Anaconda prompt: 
    '''
    conda env create -f env.yml 
    '''
    - Activate the environment
    - Run the following command: python nonstandardcode_modified.py
