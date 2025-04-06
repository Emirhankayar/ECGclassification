# PREPROCESSING PIPELINE
### Define PROJECT_DIR at constants.py.
### Data.zip, contains the data download it from Lodovico's drive.
### Unzip Data.zip, place it to project dir.
### Define WINDOW_SIZE at constants.py.
### Run pp_pipeline.py, generates the data for NNs.
### Start training model of selection

# OBTAINING RESULTS
### In Results dir, Model_Weights stands for the saved weights of the best fits.
### Each dir inside of Model_Weights has a specific name, SHORTMODELNAME_WINDOWSIZE_LAYERAMOUNT.
### If you visit RES_W500_L3 you will see residual network trained with 500,12 data and 3 residual layers.
### If you see in the Results dir, some folder containing full name of the model it is from keras tuner.

# OBTAINING BEST FIT MODELS
### src/Models dir contains all the best fit models. So far the best residual ones are T00,T01,T02,T03

# SIDE NOTE FOR LONGATO : ADD SMOTE TO THE TRAIN DATA ONLY, YOU WILL SEE HOW COOL IT IS 