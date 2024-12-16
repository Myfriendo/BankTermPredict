import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from joblib import dump

# Data loading (use your path)
path = "bank-additional/bank-additional.csv"
bank = pd.read_csv(path, sep=';')


# Preprocessing Function for Cyclic Encoding
def apply_cyclic_encoding(X):
    # Cyclic encoding for month
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    X['month'] = X['month'].map(month_mapping)
    X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
    X.drop('month', axis=1, inplace=True)

    # Cyclic encoding for day_of_week
    day_of_week_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}
    X['day_of_week'] = X['day_of_week'].map(day_of_week_mapping)
    X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 5)
    X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 5)
    X.drop('day_of_week', axis=1, inplace=True)

    return X


def apply_ordinal_encoding(X):
    # Ordinal encoding for education
    education_mapping = [['illiterate', 'unknown', 'basic.4y', 'basic.6y', 'basic.9y',
                          'high.school', 'university.degree', 'professional.course']]
    education_encoder = OrdinalEncoder(categories=education_mapping)
    X['education'] = education_encoder.fit_transform(X[['education']])

    # Ordinal encoding for default, housing, and loan
    has_loan_order = [['no', 'unknown', 'yes']]
    loan_encoder = OrdinalEncoder(categories=has_loan_order * 3)  # Repeat for three columns
    X[['default', 'housing', 'loan']] = loan_encoder.fit_transform(X[['default', 'housing', 'loan']])

    # Ordinal encoding for poutcome
    poutcome_mapping = [['failure', 'nonexistent', 'success']]  # Ordered: worst to best
    poutcome_encoder = OrdinalEncoder(categories=poutcome_mapping)
    X['poutcome'] = poutcome_encoder.fit_transform(X[['poutcome']])

    return X


def apply_label_binarizer(X):
    lb = LabelBinarizer()
    categorical_columns = ['contact', 'job', 'marital']

    # Create a DataFrame to store the transformed columns
    transformed = pd.DataFrame(index=X.index)

    for col in categorical_columns:
        binarized = lb.fit_transform(X[col])

        # If binary (e.g., two unique values), `LabelBinarizer` returns 1D array
        if binarized.shape[1] == 1:
            binarized = binarized.reshape(-1, 1)

        # Convert the binarized result into a DataFrame with proper column names
        binarized_df = pd.DataFrame(binarized, columns=[f"{col}_{cls}" for cls in lb.classes_], index=X.index)
        transformed = pd.concat([transformed, binarized_df], axis=1)

    # Drop the original categorical columns and replace them with the transformed ones
    X = X.drop(columns=categorical_columns)
    X = pd.concat([X, transformed], axis=1)
    return X


# Extract target and features
X = bank.drop(['y'], axis=1)
y = bank['y'].map({'no': 0, 'yes': 1})  # Convert 'y' to binary

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Resample training data to handle class imbalance
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Apply cyclic encoding
        ('cyclic', FunctionTransformer(apply_cyclic_encoding, validate=False), X.columns.tolist()),

        # Apply ordinal encoding
        ('ordinal', FunctionTransformer(apply_ordinal_encoding, validate=False), X.columns.tolist()),

        # Apply LabelBinarizer for categorical columns
        ('label_binarizer', FunctionTransformer(apply_label_binarizer, validate=False), X.columns.tolist()),

        # Scale numerical features
        ('num_scaler', StandardScaler(), X.select_dtypes(include='number').columns.tolist())
    ]
)

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
    'Bagging': BaggingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
    'Kernel SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

# Hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10]
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7]
    },
    'Bagging': {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_samples': [0.5, 0.75, 1.0]
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'MLP': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01]
    },
    'Linear SVM': {
        'classifier__C': [0.1, 1, 10]
    },
    'Kernel SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 0.1, 0.01]
    }
}
# Dictionary to store model results
model_results = {}

for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Get best estimator
    best_model = grid_search.best_estimator_

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    # Store results
    model_results[model_name] = {
        'model': best_model,
        'f1_score': f1,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }

    print(f"Results for {model_name}:")
    print(f"Accuracy: {model_results[model_name]['accuracy']}")
    print(f"Recall: {model_results[model_name]['recall']}")
    print(f"Precision: {model_results[model_name]['precision']}")
    print(f"F1 Score: {model_results[model_name]['f1_score']}")
    print(classification_report(y_test, y_pred))

# Choose the best model based on F1 score
best_model_name = max(model_results, key=lambda name: model_results[name]['f1_score'])
best_model = model_results[best_model_name]['model']

# Save the best model
filename = 'best_model_pipeline.sav'
pickle.dump(best_model, open(filename, 'wb'))
print(
    f"Best model saved as {filename}: {best_model_name} with F1 score of {model_results[best_model_name]['f1_score']:.4f}")
