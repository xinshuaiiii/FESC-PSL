import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix, classification_report, matthews_corrcoef
import joblib
def merge_npy_files(file_path1, file_path2):
    array1 = np.load(file_path1)
    array2 = np.load(file_path2)

    # Horizontally concatenate the arrays
    merged_array = np.hstack([array1, array2])

    return merged_array

class FASA:
    def __init__(self, num_classes, minority_class_indices, feature_dim):
        self.num_classes = num_classes
        self.minority_class_indices = minority_class_indices
        self.feature_dim = feature_dim
        self.virtual_features = np.zeros((num_classes, feature_dim))
        self.virtual_feature_prob = np.ones((num_classes, feature_dim))

    def adaptive_feature_enhancement(self, X_train, y_train):
        for idx in self.minority_class_indices:
            minority_samples = X_train[y_train[:, idx] == 1]
            self.virtual_features[idx] = np.mean(minority_samples, axis=0)

    def adaptive_feature_sampling(self, X_train, y_train):
        for idx in range(self.num_classes):
            class_samples = X_train[y_train[:, idx] == 1]
            class_size = len(class_samples)
            virtual_feature_prob_update = np.zeros(self.feature_dim)
            for i in range(self.feature_dim):
                if class_size > 0:
                    virtual_feature_prob_update[i] = np.sum(
                        class_samples[:, i] == self.virtual_features[idx, i]) / class_size

            # Check for NaN values in probability array and replace with a small non-zero value
            if np.isnan(virtual_feature_prob_update).any():
                virtual_feature_prob_update[np.isnan(virtual_feature_prob_update)] = np.finfo(float).eps

            # Add normalization to ensure sum of probabilities is 1
            total_prob = np.sum(virtual_feature_prob_update)
            if total_prob == 0:
                virtual_feature_prob_update = np.ones_like(virtual_feature_prob_update) / self.feature_dim
            else:
                virtual_feature_prob_update = virtual_feature_prob_update / total_prob

            self.virtual_feature_prob[idx] = virtual_feature_prob_update

    def oversample_minority_samples(self, X_train, y_train, oversample_ratio=3):
        X_resampled = X_train
        y_resampled = y_train
        for idx in self.minority_class_indices:
            minority_samples = X_train[y_train[:, idx] == 1]
            num_samples = len(minority_samples)
            num_virtual_samples = int(num_samples * oversample_ratio)
            virtual_feature_indices = np.random.choice(self.feature_dim, size=num_virtual_samples,
                                                       p=self.virtual_feature_prob[idx])
            virtual_samples = np.array([self.virtual_features[idx] for _ in range(num_virtual_samples)])
            X_resampled = np.vstack([X_resampled, virtual_samples])
            y_resampled = np.vstack([y_resampled, np.tile(y_train[idx], (num_virtual_samples, 1))])
        return X_resampled, y_resampled

file_path1 = 'psepssm.npy'
file_path2 = 'prott5.npy'
X = merge_npy_files(file_path1, file_path2)
print("Shape of merged array:", X.shape)

with open('label', 'r') as f:
    y_str = f.readlines()

y = np.array([list(map(int, line.strip()[1:-1].split())) for line in y_str])

# Initialize FASA
fasa = FASA(num_classes=4, minority_class_indices=[2,3], feature_dim=X.shape[1])

# Define hyperparameters to tune, adjusted for nested estimator
svm_param_grid = {
    'estimator__C': [1, 10, 100],  
    'estimator__gamma': ['scale', 'auto'，'rbf']
}


# Initialize cross-validation and GridSearchCV
overall_precision_list, overall_recall_list, overall_mcc_list = [], [], []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_categorical = np.argmax(y, axis=1)

# Inside your loop for stratified cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(X, y_categorical)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(1)
    # FASA Processing
    fasa.adaptive_feature_enhancement(X_train, y_train)
    fasa.adaptive_feature_sampling(X_train, y_train)
    X_train_resampled, y_train_resampled = fasa.oversample_minority_samples(X_train, y_train)

    # Set up SVM classifier within a MultiOutputClassifier
    svm_classifier = SVC()
    multi_output_classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)  # n_jobs=-1 to use all processors

    # Set up GridSearchCV with correct parameter prefix
    grid_search = GridSearchCV(multi_output_classifier, param_grid=svm_param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Evaluate and print
    y_pred = grid_search.best_estimator_.predict(X_test)
    y_pred_binary = np.round(y_pred)
    mcc_per_label = [matthews_corrcoef(y_test[:, i], y_pred_binary[:, i]) for i in range(y.shape[1])]
    overall_mcc_list.append(mcc_per_label)
    overall_precision_list.append(classification_report(y_test, y_pred_binary, output_dict=True)['macro avg']['precision'])
    overall_recall_list.append(classification_report(y_test, y_pred_binary, output_dict=True)['macro avg']['recall'])

    print(f"Fold {fold+1} Best Params: {grid_search.best_params_}")
    print(f"Classification report for Fold {fold+1}:\n{classification_report(y_test, y_pred_binary)}")


# Save the best model
joblib.dump(grid_search.best_estimator_, 'best_svm_fasa_model.pkl')

# Display averaged results
mean_overall_precision, mean_overall_recall = np.mean(overall_precision_list), np.mean(overall_recall_list)
mean_mcc_per_label = np.mean(overall_mcc_list, axis=0)
print(f"Overall Mean Precision: {mean_overall_precision}")
print(f"Overall Mean Recall: {mean_overall_recall}")
print(f"Mean MCC per label: {mean_mcc_per_label}")
