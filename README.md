# README
This file contains the instructions for running the project.

Once the Knee MRI Dataset is downloaded, run the MRNet(1).ipynb notebook file preprocess the data and save the numpy arrays of the preprocessed data.

then run the balancing_dataset_stylegan.ipynb notebook to load the preprocessed numpy arrays and store them in there seperate folders. It then also balances the data and creates a dataset.json file in each folder which is required to create class conditional images in stylegan2 and stylegan2 ada.

Next import and run stylegan2/stylegan2-ADA on the class seperated images. whcih is in the same notebook

To train the GAN models it requires the path to image folders which were created using balancing_dataset_stylegan.ipynb which 

the trained models and the notebooks for the 2 CNN models on the original data and the synthetic data can be found on the following drive link https://qmulprod-my.sharepoint.com/:f:/g/personal/ec211128_qmul_ac_uk/EsAhBeuVAalAkUiYqyIFvlgBTuWfvkDYawX6vNXA8yctlA?e=3iQzlT.

THE FID and IS  scores notebook can be calculated by running the notebooks specified in the metrics folder for each model. we use the trainind models to generate images and then calculate the FID and IS scores. 1 folder is needed, a folder containing the original images from the preprocessed MRnet dataset, which can be made using the balancing_dataset notebook. Then change the path of the trained model in the notebook to your own path and this will calculate the FID and IS scores on this data. This is for each model CGAN, DCGAN, Stylegan2 and Stylegan 2 ADA.
