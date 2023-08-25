import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="auto",
)


# Load the trained model
model = load_model('model.h5')

# Load the tokenizer
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open('tokenizer.json').read())

# Load the label encoder
label_encoder_classes = np.load('label_encoder.npy')
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Map sentiment class values to labels
sentiment_labels = {
    0: "Your Sentiment is Negative",
    1: "Your Sentiment is Positive",
    2: "Your Sentiment is Neutral"
}


# Streamlit app
st.title("Sentiment Analysis App")
st.markdown("---")


# Add a textarea for user input
user_input = st.text_area("Enter your review:", "")

# Add a prediction button
if st.button("Predict", key="predict_button", ):
    if user_input:
        # Tokenize and pad the user input
        user_input_sequence = tokenizer.texts_to_sequences([user_input])
        user_input_padded = tf.keras.preprocessing.sequence.pad_sequences(
            user_input_sequence,
            maxlen=40,
            padding='post',
            truncating='post'
        )

        # Predict sentiment
        prediction = model.predict(user_input_padded)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map the predicted class to sentiment label using the dictionary
        sentiment_label = sentiment_labels.get(predicted_class, "unknown")

        # Display the predicted sentiment
        st.title(f"Predicted Sentiment: {sentiment_label}")
    else:
        st.warning("Please enter a review.")

# Add whitespace
st.markdown("---")

# Footer
st.write("Made with ❤️ by ByteSifters")