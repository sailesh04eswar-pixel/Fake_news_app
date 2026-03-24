import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📰 Fake News Detector")

# Input box
text = st.text_area("Enter news text:")

# Button
if st.button("Check News"):
    
    if text.strip() == "":
        st.warning("Please enter some text")
    
    else:
        # Transform input
        input_data = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)
        
        # Confidence
        confidence = max(prob[0]) * 100
        
        # Output
        if prediction[0] == 1:
            st.success(f"Real News ✅ ({confidence:.2f}%)")
        else:
            st.error(f"Fake News ❌ ({confidence:.2f}%)")
