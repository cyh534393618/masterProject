import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
import joblib
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.utils import plot_model

# Load a trained model
def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load validated.tsv file
def load_validated_data(tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t')

    # filter conditions
    filtered_df = df[
        df['gender'].notna() & (df['gender'] != '') &                  # 性别不为空
        (df['gender'] != 'other') &                                     # 排除 'other'
        (df['up_votes'] > 3) &                                          # up_votes 大于 3
        (df['down_votes'] == 0) &                                       # down_votes 等于 0
        df['age'].notna() & (df['age'] != '') &                        # age 不为空
        df['accent'].notna() & (df['accent'] != '')                    # accent 不为空
    ]

    print(filtered_df.shape)

    return filtered_df

def load_new_data(tsv_file):
    df = load_validated_data(tsv_file)
    features = []
    accents = []
    ages = []
    genders = []

    for index, row in df.iterrows():
        audio_path = os.path.join('cv-corpus-18.0-2022-06-14', 'en\clips', row['path'])

        # Check file format, if it is MP3 convert to WAV
        if audio_path.endswith('.mp3'):
            audio_path = audio_path.replace('.mp3', '.wav')
            #audio_path = convert_mp3_to_wav(audio_path)

        # Read audio files (now guaranteed to be in WAV format)
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
        mfccs = np.mean(mfccs.T, axis=0)
        features.append(mfccs)
        
        accents.append(row['accent'])
        ages.append(row['age'])
        genders.append(row['gender'])

    features_array = np.array(features)
    accents_array = np.array(accents)
    ages_array = np.array(ages)
    genders_array = np.array(genders)

    return features_array, accents_array, ages_array, genders_array

def create_cnn_model(input_shape, num_accents, num_ages, num_genders):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Reshape((30, 1))(inputs)

    # convolution layer
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)

    # shared layer
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Accent output
    accent_output = tf.keras.layers.Dense(num_accents, activation='softmax', name='accent')(x)

    # Age output
    age_output = tf.keras.layers.Dense(num_ages, activation='softmax', name='age')(x)

    # Gender output
    gender_output = tf.keras.layers.Dense(num_genders, activation='softmax', name='gender')(x)

    # Define model
    model = tf.keras.Model(inputs=inputs, outputs=[accent_output, age_output, gender_output])
    
    return model

def train_model(tsv_file):
    X_new, accents, ages, genders = load_new_data(tsv_file)

    # Label encoding
    le_accent = LabelEncoder()
    le_age = LabelEncoder()
    le_gender = LabelEncoder()

    y_accent_encoded = le_accent.fit_transform(accents)
    y_age_encoded = le_age.fit_transform(ages)
    y_gender_encoded = le_gender.fit_transform(genders)

    X_train, X_test, y_accent_train, y_accent_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
        X_new, y_accent_encoded, y_age_encoded, y_gender_encoded, test_size=0.2, random_state=42)

    model = create_cnn_model((X_train.shape[1],), len(le_accent.classes_), len(le_age.classes_), len(le_gender.classes_))
    model.compile(optimizer='adam', 
                  loss={'accent': 'sparse_categorical_crossentropy', 
                        'age': 'sparse_categorical_crossentropy', 
                        'gender': 'sparse_categorical_crossentropy'},
                  metrics={
                      'accent': 'accuracy',
                      'age': 'accuracy',
                      'gender': 'accuracy'
                  })

    model.fit(X_train, 
              {'accent': y_accent_train, 'age': y_age_train, 'gender': y_gender_train},
              epochs=30, 
              batch_size=32, 
              validation_split=0.2)

    # Model evaluation
    results = model.evaluate(
        X_test, 
        {
            'accent': y_accent_test, 
            'age': y_age_test, 
            'gender': y_gender_test
        }
    )

    # Calculate the loss and accuracy of each task based on the output results
    # Make sure the result is the correct length
    num_outputs = len(results)
    if num_outputs == 6:  # 包含损失和准确率
        accent_loss, age_loss, gender_loss, accent_accuracy, age_accuracy, gender_accuracy = results

        print(f'Test loss: Accent: {accent_loss:.4f}, Age: {age_loss:.4f}, Gender: {gender_loss:.4f}')
        print(f'Test accuracy: Accent: {accent_accuracy:.2f}, Age: {age_accuracy:.2f}, Gender: {gender_accuracy:.2f}')

    # Save encoder
    joblib.dump(le_accent, 'le_accent.pkl')
    joblib.dump(le_age, 'le_age.pkl')
    joblib.dump(le_gender, 'le_gender.pkl')
    
    model.save('sound_classification_model.keras')

    print(f'Evaluate model:')
    evaluate_model(model, X_test, y_accent_test, y_age_test, y_gender_test, (le_accent, le_age, le_gender))

    # Draw the model architecture diagram and save it as a PNG file
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True, dpi=300)

    # Print a summary of the model architecture
    model.summary()

    return le_accent, le_age, le_gender

# Test a single WAV file
def test_single_file(model, file_path, encoders):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    mfccs = np.mean(mfccs.T, axis=0).reshape(1, -1)

    predictions = model.predict(mfccs)
    predicted_accent = encoders[0].inverse_transform([np.argmax(predictions[0])])
    predicted_age = encoders[1].inverse_transform([np.argmax(predictions[1])])
    predicted_gender = encoders[2].inverse_transform([np.argmax(predictions[2])])

    print(f"Predicted accent: {predicted_accent[0]}")
    print(f"Predicted age: {predicted_age[0]}")
    print(f"Predicted gender: {predicted_gender[0]}")

def fine_tune_model(model, tsv_file, encoders, model_path, epochs=10):
    X_new, accents, ages, genders = load_new_data(tsv_file)

    # Tag encoding
    # y_accent_encoded = encoders[0].transform(accents)
    # y_age_encoded = encoders[1].transform(ages)
    # y_gender_encoded = encoders[2].transform(genders)

    # Label encoding
    le_accent = LabelEncoder()
    le_age = LabelEncoder()
    le_gender = LabelEncoder()

    y_accent_encoded = le_accent.fit_transform(accents)
    y_age_encoded = le_age.fit_transform(ages)
    y_gender_encoded = le_gender.fit_transform(genders)

    X_train, X_test, y_accent_train, y_accent_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
        X_new, y_accent_encoded, y_age_encoded, y_gender_encoded, test_size=0.2, random_state=42)

    # Unfreeze some layers for fine-tuning
    # for layer in model.layers[-2:]:
    #     layer.trainable = True

    model = create_cnn_model((X_train.shape[1],), len(le_accent.classes_), len(le_age.classes_), len(le_gender.classes_))
    model.compile(optimizer='adam',
                  loss={'accent': 'sparse_categorical_crossentropy', 
                        'age': 'sparse_categorical_crossentropy', 
                        'gender': 'sparse_categorical_crossentropy'},
                  metrics={
                      'accent': 'accuracy',
                      'age': 'accuracy',
                      'gender': 'accuracy'
                  })
    
    model.fit(X_train, 
              {'accent': y_accent_train, 'age': y_age_train, 'gender': y_gender_train},
              epochs=epochs, 
              batch_size=32, 
              validation_split=0.2)

    results = model.evaluate(X_test, 
                            {
                                'accent': y_accent_test, 
                                'age': y_age_test, 
                                'gender': y_gender_test
                            })
    
    # Calculate the loss and accuracy of each task based on the output results
    # Make sure the result is the correct length
    num_outputs = len(results)
    if num_outputs == 6:
        accent_loss, age_loss, gender_loss, accent_accuracy, age_accuracy, gender_accuracy = results

        print(f'Test loss: Accent: {accent_loss:.4f}, Age: {age_loss:.4f}, Gender: {gender_loss:.4f}')
        print(f'Test accuracy: Accent: {accent_accuracy:.2f}, Age: {age_accuracy:.2f}, Gender: {gender_accuracy:.2f}')

    # Save encoder
    joblib.dump(le_accent, 'le_accent.pkl')
    joblib.dump(le_age, 'le_age.pkl')
    joblib.dump(le_gender, 'le_gender.pkl')
    
    # Resave the fine-tuned model
    model.save(model_path)
    print(f'Model fine-tuned and saved at: {model_path}')

def save_new_data(path, voicePath, gender, up_votes, down_votes, age, accent):
    # New data example
    new_data = {
        'path': [voicePath],
        'gender': [gender],
        'up_votes': [up_votes],
        'down_votes': [down_votes],
        'age': [age],
        'accent': [accent]
    }

    # Create a new DataFrame
    new_df = pd.DataFrame(new_data)

    # Read existing TSV file
    existing_file = path
    existing_df = pd.read_csv(existing_file, delimiter='\t')

    # Merge new data
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save the updated TSV file
    updated_df.to_csv(existing_file, sep='\t', index=False)
    print(f"Updated TSV file saved with {len(updated_df)} entries.")

def evaluate_model(model, X_test, y_accent_test, y_age_test, y_gender_test, encoders):
    # 进行预测
    predictions = model.predict(X_test)

    # 提取每个输出的预测结果
    y_accent_pred = np.argmax(predictions[0], axis=1)
    y_age_pred = np.argmax(predictions[1], axis=1)
    y_gender_pred = np.argmax(predictions[2], axis=1)

    # 计算准确率
    accent_accuracy = accuracy_score(y_accent_test, y_accent_pred)
    age_accuracy = accuracy_score(y_age_test, y_age_pred)
    gender_accuracy = accuracy_score(y_gender_test, y_gender_pred)

    # 计算 F1 分数
    accent_f1 = f1_score(y_accent_test, y_accent_pred, average='weighted', labels=np.unique(y_accent_pred))
    age_f1 = f1_score(y_age_test, y_age_pred, average='weighted', labels=np.unique(y_age_pred))
    gender_f1 = f1_score(y_gender_test, y_gender_pred, average='weighted', labels=np.unique(y_gender_pred))

    print(f'Accent Accuracy: {accent_accuracy:.2f}, F1 Score: {accent_f1:.2f}')
    print(f'Age Accuracy: {age_accuracy:.2f}, F1 Score: {age_f1:.2f}')
    print(f'Gender Accuracy: {gender_accuracy:.2f}, F1 Score: {gender_f1:.2f}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio Processing Model")
    parser.add_argument('--mode', choices=['fine-tune', 'train', 'test'], required=True, help='Mode of operation')
    parser.add_argument('--model-path', type=str, help='Path to the model file (for fine-tuning and testing)')
    parser.add_argument('--data-path', type=str, help='Path to the TSV file (for training and fine-tuning)')
    parser.add_argument('--test-file', type=str, help='Path to the audio file to test (for testing mode)')
    parser.add_argument('--voice-path', type=str, help='Path to the audio file (for fine-tuning)')
    parser.add_argument('--gender', type=str, help='Gender (for fine-tuning)')
    parser.add_argument('--age', type=str, help='Age (for fine-tuning)')
    parser.add_argument('--accent', type=str, help='Accent (for fine-tuning)')

    args = parser.parse_args()

    if args.mode == 'fine-tune':
        # Train the model and get the label encoder
        model = load_trained_model(args.model_path)

        # Save latest data
        save_new_data(args.data_path, args.voice_path, args.gender, 5, 0, args.age, args.accent)

        # Load encoder
        le_accent = joblib.load('le_accent.pkl')
        le_age = joblib.load('le_age.pkl')
        le_gender = joblib.load('le_gender.pkl')
        encoders = (le_accent, le_age, le_gender)

        fine_tune_model(model, args.data_path, encoders, args.model_path, 30)

    elif args.mode == 'train':
        train_model(args.data_path)
    
    elif args.mode == 'test':
        model = load_trained_model(args.model_path)
        
        # Load encoder
        le_accent = joblib.load('le_accent.pkl')
        le_age = joblib.load('le_age.pkl')
        le_gender = joblib.load('le_gender.pkl')
        encoders = (le_accent, le_age, le_gender)

        # Test a single WAV file
        test_single_file(model, args.test_file, encoders)
        