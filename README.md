# FESC-PSL
Feature Exaction：

  To generate ProtT5 features, we need to download the ProtT5 model from https://github.com/agemagician/ProtTrans and set up the corresponding tokenizer and model path in the 'prott5-feature.py' file. The generated feature file will be named 'prott5.npy'. 
  To generate PsePSSM features, we first need to create PSSM files and save them in a folder. We then adjust parameters based on the number of PSSM files to generate the 'psepssm.npy' feature file.
  
Train and Test:

  Set the prott5-feature.py and psepssm.npy respectively to the file1 and file2 paths in train.py to proceed with training and testing.
