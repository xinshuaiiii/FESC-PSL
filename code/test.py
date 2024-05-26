import numpy as np
from sklearn.metrics import f1_score, jaccard_score, hamming_loss
from joblib import load
from sklearn.preprocessing import RobustScaler
def merge_npy_files(file_path1, file_path2):
    array1 = np.load(file_path1)
    array2 = np.load(file_path2)

    # Horizontally concatenate the arrays
    merged_array = np.hstack([array1, array2])

    return merged_array

file_path1 = 'psepssm.npy'
file_path2 = 'prott5.npy'
X_test = merge_npy_files(file_path1, file_path2)
print("Shape of merged array:", X_test.shape)
# Load the test features and labels
with open('label.txt', 'r') as f:
    y_str = f.readlines()

y_test = np.array([list(map(int, line.strip()[1:-1].split())) for line in y_str])


# Load the trained model
model = load('model.pkl')  

# Generate predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
micro_f1 = f1_score(y_test, y_pred, average='micro')
macro_f1 = f1_score(y_test, y_pred, average='macro')
jaccard = jaccard_score(y_test, y_pred, average='samples')  # 'samples' averages Jaccard over the sample set
hl = hamming_loss(y_test, y_pred)

# Calculate F1-score for each label
label_f1_scores = f1_score(y_test, y_pred, average=None)

# Output performance metrics
print("Micro-F1:", micro_f1)
print("Macro-F1:", macro_f1)
print("Jaccard:", jaccard)
print("Hamming Loss:", hl)
print("F1-score for each label:", label_f1_scores)
