import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib


def merge_npy_files(file_path1, file_path2):
    array1 = np.load(file_path1)
    array2 = np.load(file_path2)
    merged_array = np.hstack([array1, array2])
    return merged_array


def normalize_probabilities(probs):
    min_prob = np.min(probs)
    max_prob = np.max(probs)
    normalized_probs = (probs - min_prob) / (max_prob - min_prob)
    return normalized_probs


class FASA:
    def __init__(self, num_classes, minority_class_indices, feature_dim):
        self.num_classes = num_classes
        self.minority_class_indices = minority_class_indices
        self.feature_dim = feature_dim
        self.virtual_features = np.zeros((num_classes, feature_dim))
        self.virtual_feature_prob_unnormalized = np.ones((num_classes, feature_dim))

    def adaptive_feature_enhancement(self, X_train, y_train):
        for idx in self.minority_class_indices:
            minority_samples = X_train[y_train[:, idx] == 1]
            if len(minority_samples) > 0:
                self.virtual_features[idx] = np.mean(minority_samples, axis=0)

    def adaptive_feature_sampling(self, X_train, y_train):
        epsilon = np.finfo(float).eps
        for idx in self.minority_class_indices:
            class_samples = X_train[y_train[:, idx] == 1]
            if len(class_samples) > 0:
                distances = np.abs(class_samples - self.virtual_features[idx])
                mean_distances = distances.mean(axis=0)
                virtual_feature_prob_update = 1.0 / (mean_distances + epsilon)
                self.virtual_feature_prob_unnormalized[idx] = virtual_feature_prob_update

    def oversample_minority_samples(self, X_train, y_train, oversample_ratio=3):
        X_resampled = np.copy(X_train)
        y_resampled = np.copy(y_train)
        for idx in self.minority_class_indices:
            minority_samples = X_train[y_train[:, idx] == 1]
            minority_labels = y_train[y_train[:, idx] == 1]
            num_samples = len(minority_samples)
            num_virtual_samples = int(num_samples * oversample_ratio)
            for _ in range(num_virtual_samples):
                virtual_sample = self.generate_virtual_sample(minority_samples, idx, num_samples)
                X_resampled = np.vstack([X_resampled, virtual_sample])
                virtual_label = self.generate_virtual_label(minority_labels)
                y_resampled = np.vstack([y_resampled, virtual_label])
        return X_resampled, y_resampled

    def generate_virtual_sample(self, minority_samples, idx, num_samples):
        virtual_sample = np.zeros(self.feature_dim)
        normalized_probs = normalize_probabilities(self.virtual_feature_prob_unnormalized[idx])
        for feature_idx in range(self.feature_dim):
            prob_threshold = normalized_probs[feature_idx]
            if np.random.rand() <= prob_threshold:
                virtual_sample[feature_idx] = self.virtual_features[idx][feature_idx] + np.random.normal(0, 0.1)
            else:
                sample_choice1 = np.random.randint(num_samples)
                sample_choice2 = np.random.randint(num_samples)
                alpha = np.random.rand()
                virtual_sample[feature_idx] = alpha * minority_samples[sample_choice1, feature_idx] + (1 - alpha) * \
                                              minority_samples[sample_choice2, feature_idx]
        return virtual_sample

    def generate_virtual_label(self, minority_labels):
        num_samples = len(minority_labels)
        sample_choice1 = np.random.randint(num_samples)
        sample_choice2 = np.random.randint(num_samples)
        alpha = np.random.rand()
        virtual_label = alpha * minority_labels[sample_choice1] + (1 - alpha) * minority_labels[sample_choice2]
        virtual_label = (virtual_label > 0.5).astype(int)
        return virtual_label


file_path2 = 'seq-prott5.npy'
with open('label.txt', 'r') as f:
    y_str = f.readlines()
y = np.array([list(map(int, line.strip()[1:-1].split())) for line in y_str])

svm_classifier = SVC(kernel='rbf', C=10, gamma='scale')
y_categorical = np.argmax(y, axis=1)

for lamda in range(1, 11):
    file_path1 = f'psepssm-lamda{lamda}.npy'
    X = merge_npy_files(file_path1, file_path2)
    print(f"Processing lamda {lamda}, Shape of merged array:", X.shape)

    # Initialize FASA with the correct feature_dim
    fasa = FASA(num_classes=5, minority_class_indices=[2,3,4], feature_dim=X.shape[1])

    best_f1 = -1  
    best_model = None  
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y_categorical)):
        print(f"Fold {fold + 1}:")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        fasa.adaptive_feature_enhancement(X_train, y_train)
        fasa.adaptive_feature_sampling(X_train, y_train)
        X_train_resampled, y_train_resampled = fasa.oversample_minority_samples(X_train, y_train)


        multi_output_classifier = MultiOutputClassifier(svm_classifier)
        multi_output_classifier.fit(X_train_resampled, y_train_resampled)


        y_pred = multi_output_classifier.predict(X_test)


        macro_f1 = f1_score(y_test, y_pred, average='macro')


        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = multi_output_classifier


    joblib.dump(best_model, f'lamda{lamda}.pkl')
    print(f"best lamda {lamda}，Macro F1 :", best_f1)
