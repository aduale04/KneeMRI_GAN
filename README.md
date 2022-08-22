# README
This file contains the instructions for running the project.

Once the Knee MRI Dataset is downloaded, run the MRNet(1).ipynb notebook file preprocess the data and save the numpy arrays of the preprocessed data.

then run the balancing_dataset_stylegan.ipynb notebook to load the preprocessed numpy arrays and store them in there seperate folders. It then also balances the data and creates a dataset.json file in each folder which is required to create class conditional images in stylegan2.

Next import and run stylegan2/stylegan2-ADA on the class seperated images and 
