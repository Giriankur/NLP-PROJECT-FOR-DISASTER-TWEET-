import streamlit as st
import pickle

# Load the saved model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    return model, tfidf

# Function to make predictions
def predict(text, model, tfidf):
    text_tfidf = tfidf.transform([text])  # Transform text using the vectorizer
    prediction = model.predict(text_tfidf)  # Get prediction
    return int(prediction[0])  # Convert to int for consistency

# Streamlit app UI
st.title("Disaster Tweet Classification")
st.subheader("Predict whether a tweet is related to a disaster or not.")

# Load model and vectorizer
model, tfidf = load_model_and_vectorizer()

# Input from user
user_input = st.text_input("Enter a tweet to classify:", placeholder="e.g., Flooding in California is devastating communities.")

# Predict button
if st.button("Classify"):
    if user_input.strip():
        result = predict(user_input, model, tfidf)
        category = "Disaster" if result == 1 else "Non-Disaster"
        st.success(f"The tweet is classified as: **{category}**")
    else:
        st.error("Please enter a valid tweet.")

# Sidebar information
st.sidebar.title("About This App")
st.sidebar.info(
    """
    This application uses a machine learning model to classify tweets as either:
    - **Disaster**: Related to natural or man-made disasters.
    - **Non-Disaster**: Not related to disasters.
    
    **Note**: The model may not always provide accurate results.
    """
)

with open('models/model.pkl', 'rb') as file:
    
with open(r'C:\Users\272530\Desktop\Streamlit_07\model.pkl', 'rb') as file:
    
        
