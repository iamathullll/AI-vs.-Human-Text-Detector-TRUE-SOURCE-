import streamlit as st
import pickle
import numpy as np

with open('AINEW2savedobjects.pkl', "rb") as file:  
    data_dict = pickle.load(file)

def predict_generated(text, data_dict):
    nlp = data_dict["NLP"]
    model = data_dict["Model"]
    vectorizer = data_dict["vectorizer"]

    doc = nlp(text.lower())
    processed_text = " ".join([token.text for token in doc])
    word_count = len([token for token in doc if not token.is_punct])
    sentence_count = len(list(doc.sents))
    avg_word_length = (
        sum(len(token.text) for token in doc if not token.is_punct) / word_count
        if word_count > 0 else 0
    )
    punctuation_count = len([token for token in doc if token.is_punct])
    stop_word_count = len([token for token in doc if token.is_stop])
    unique_word_count = len(set(token.text for token in doc if not token.is_punct))
    lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
    
    additional_features = np.array([[
        word_count, sentence_count, avg_word_length, 
        punctuation_count, stop_word_count, 
        unique_word_count, lexical_diversity
    ]])
    
    X_text = vectorizer.transform([processed_text]).toarray()
    
    combined_features = np.hstack((X_text, additional_features))
    
    prediction = model.predict(combined_features)[0]
    
    return "AI Generated" if prediction == 1 else "Human Generated"

st.title("True Source : AI vs Human Text Detector")
st.write("Enter a piece of text to determine if it's AI-generated or written by a human.")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Analyze"):
    if user_input.strip():
        result = predict_generated(user_input, data_dict)
        st.subheader("Result")
        st.write(f"The text is **{result}**")
    else:
        st.warning("Please enter some text to analyze.")
# streamlit run "/Users/athulkrishnavv/Desktop/ML&DL/AI vs human text/TrueSource-.py"