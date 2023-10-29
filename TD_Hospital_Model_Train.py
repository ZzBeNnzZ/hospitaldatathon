import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow import keras
from keras import layers


def data_preprocessing(df):
    
    col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    df = df[col_to_keep]

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    return df
    
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
    df['Male'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['Female'] = df['sex'].apply(lambda x: 1 if x == 'female' else 0)
    df.drop('sex', axis=1, inplace=True)
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
    
    # If the percentage is low,  drop those rows altogether
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
    df_encoded = pd.get_dummies(df, columns=['sex', 'race', 'dnr', 'primary', 'disability','income', 'extraprimary', 'cancer'], prefix=['sex', 'race', 'dnr','primary',  'disability','income', 'extraprimary', 'cancer'])

    return df_encoded

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Explicitly convert to float32
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

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


    # # Optionally, you can plot training history to visualize model performance
    # import matplotlib.pyplot as plt

    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0, 1])
    # plt.legend(loc='lower right')
    # plt.show()



if __name__ == "__main__":
    data_path = 'C:\\Users\\benle\\OneDrive\\BenLeTAMU\\TAMUDATATHON\\TDHospital\\TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    df.head()
    df = handle_missing_data(df)
    # df = data_preprocessing(df)


    # Standardize sex value:
    df = encodeData(df)

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print("Non-numeric columns:", non_numeric_cols)
    

    # Calculate the correlation with the column death
    correlation_with_death = df.corr()['death']

    # TO DO: incorporate correlation with training model
    # Select features based on correlation threshold
    threshold = 0.2
    relevant_features = correlation_with_death[correlation_with_death.abs() > threshold].index.tolist()

    # Ensure 'death' is not in the list of relevant features
    relevant_features.remove('death')

    
    y, X = split_feature_label(df)
    X = standardize(X)
    train_model(X, y)

    # #Changing the sex column up and turning it to 0 and 1
    # divide_sex_column(df)

    # #race mapping:
    # race_mapping(df)

    
    # # print("Correlation variable: " + correlation_with_death)
    
    # columns = df.columns
    # print(columns)