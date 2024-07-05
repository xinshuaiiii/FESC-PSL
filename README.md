# FESC-PSL
For the construction and principle of the model, please refer to FESC-PSL: Bacterial Protein Subcellular Localization Prediction Based on Pre-trained Protein Language Model and FASA (submission in progress).

If you have any questions, please contact: liuyun313@jlu.edu.cn or xinshaui23@jlu.edu.cn

# Creating a Virtual Environment
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies.The command is as follows：
```
conda create -n predict pyhton=3.7.13
conda activate predict
git clone https://github.com/xinshuaiiii/FESC-PSL.git
cd FESC-PSL
pip install -r requirements.txt
```

#Feature Exaction：
  To generate ProtT5 features, we need to download the ProtT5 model from https://github.com/agemagician/ProtTrans and set up the corresponding tokenizer and model path in the 'prott5-feature.py' file. The generated feature file will be named 'prott5.npy'.Then run: 
  
```
python prott5-feature.py
```

  To generate PsePSSM features, we first need to create PSSM files and save them in a folder. We then adjust parameters based on the number of PSSM files to generate the 'psepssm.npy' feature file.Then run:

```
python psepssm.py
```

  
Train and Test:

  Set the prott5-feature.py and psepssm.npy respectively to the file1 and file2 paths in train.py to proceed with training and testing.
