# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

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

- Open Ubuntu terminal
- To setup Python environment run: ```conda env -f env.yml```
- To activate the Environment run: ```conda activate mle-dev```
- To Run the script: ```python nonstandardcode.py```
- Install package mle_training:  ```pip install dist/mle_training-0.1.0-py3-none-any.whl```
- Run main.py python script:  ```cd scripts```  ```python3 main.py ```
- Run basic tests:  ```cd ../tests```  ```python3 -m pytest tests.py```

## To build the Docker container and run the script

- Open Ubuntu terminal and navigate to the mle_training folder
- Build the docker image by runnning : ```docker build -t mle_training . ```
- Run the docker container by running : ```docker run -it mle_training /bin/bash```
-  Docker will start a new container from the mle_training image and attach an interactive bash shell to it, allowing you to interact with the container's file system and execute commands inside the container
- Activate the environment to run the main script by running : ```conda activate mle-dev```
- Install the package mle_training-0.1.0 by running : ```pip install dist/mle_training-0.1.0-py3-none-any.whl```
- Run main.py python script:  ```cd scripts```  ```python3 main.py ```
- Run basic tests:  ```cd ../tests```  ```python3 -m pytest tests.py```
