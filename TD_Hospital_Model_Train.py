import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras import layers
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def data_preprocessing(df):
    
    col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    df = df[col_to_keep]

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def standardize_sex(value):
    if value in ["Male", "male", "m", "M", "1"]:
        return "Male"
    elif value in ["Female", "female", "f", "F", "0"]:
        return "Female"
    else:
        return value  # or return some default value or raise an error

def divide_sex_column(df):
    df['Male'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Female'] = df['sex'].apply(lambda x: 1 if x == 'Female' else 0)
    return df

def race_mapping(df):
    # Define the mapping
    race_mapping = {
        'white': 0,
        'black': 1,
        'asian': 2,
        'hispanic': 3,}
    
    # Apply the mapping to the 'race' column
    df['race'] = df['race'].replace(race_mapping)

def percent_row_with_null (df ):
    df = df.drop(columns=["pdeath"])

    # Calculate the number of rows with at least one null value
    num_rows_with_nulls = df.isnull().any(axis=1).sum()
    
    # Calculate the percentage
    percent = (num_rows_with_nulls / len(df)) * 100
    
    return percent

def handle_missing_data(df):
    threshold=0.5
    # Calculate the percentage of rows with missing values
    total_rows = len(df)
    rows_with_missing = df.isnull().any(axis=1).sum()
    percent_missing = (rows_with_missing / total_rows) * 100
    
    print(f"Percentage of rows with missing values: {percent_missing:.2f}%")
    
    # If the percentage is low, drop those rows altogether
    if percent_missing < 5:  
        df = df.dropna()
        return df
    
    # For rows with more than 'threshold' proportion of missing values, drop them
    mask = df.isnull().mean(axis=1) > threshold
    df = df[~mask]
    
    # For the remaining rows, fill missing values with 0 
    df = df.fillna(0)
    
    return df

# One hot encode column with words to variable to train model
def encodeData (df):
    df_encoded = pd.get_dummies(df, columns=['sex', 'race', 'dnr', 'primary', 'disability','income', 'extraprimary', 'cancer'], prefix=['sex', 'race', 'dnr', 'primary', 'disability','income', 'extraprimary', 'cancer'])

    return df_encoded

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    return accuracy

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(128, activation='relu'),     # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(64, activation='relu'),      # Another hidden layer with 64 neurons and ReLU activation
        layers.Dense(1, activation='sigmoid')     # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.h5')
    
    print(f'Test accuracy: {test_accuracy}')


    # Optionally, you can plot training history to visualize model performance
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    
def train_random_forest(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Test accuracy: {accuracy}')

def train_svm(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)
    
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM Test accuracy: {accuracy}')

def train_regression(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Regression Test accuracy: {accuracy}')

def apply_oversampling(X, y):
    # Apply oversampling using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def apply_pca(X):
    # Create a PCA instance with the specified number of components
    pca = PCA(n_components=74)
    # Fit and transform the feature data
    X_pca = pca.fit_transform(X)
    return X_pca


if __name__ == "__main__":
    data_path = './TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    df.head()
    # df = data_preprocessing(df)
    df = handle_missing_data(df)

    df['sex'] = df['sex'].apply(standardize_sex)
    # #Changing the sex column up and turning it to 0 and 1
    divide_sex_column(df)

    # #race mapping:
    race_mapping(df)

    # Standardize sex value:
    df = encodeData(df)

    # non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # print("Non-numeric columns:", non_numeric_cols)
    
    y, X = split_feature_label(df)
    X = standardize(X)
    # train_model(X, y)
    # train_random_forest(X, y)
    # train_svm(X, y)
    # train_regression(X, y)

    # Apply pcs to reduce dimensionality
    X_pca = apply_pca(X)

    X_resampled, y_resampled = apply_oversampling(X_pca, y)
    
    # Train your model with the resampled data
    train_model(X_resampled, y_resampled)

    # data_features = ['sex_Male', 'sex_Female', 'race_0', 'race_1', 'race_2', 'race_3']
    # a = df[data_features]
    # print(a)