# AI-vs.-Human-Text-Detector-TRUE-SOURCE-
# **True News Validator**

## **Overview**
*True News Validator* is an interactive web application designed to classify news articles as "Real" or "Fake" using Natural Language Processing (NLP) and Machine Learning. The project aims to combat misinformation by providing an accurate and accessible fake news detection tool.

---

## **Features**
- **High Accuracy**: Achieved **97% accuracy** using Logistic Regression for classification.
- **Real-Time Classification**: Users can input news text and receive instant feedback.
- **Advanced NLP Pipeline**: Preprocessed text using SpaCy for lemmatization, stopword removal, and punctuation filtering.
- **TF-IDF Vectorization**: Transformed unstructured text data into numerical features optimized for machine learning.
- **Streamlit Integration**: Built a user-friendly web interface for seamless interaction.

---

## **Technical Stack**
- **Programming Language**: Python
- **Libraries and Tools**:
  - `pandas`, `NumPy`: Data handling and analysis
  - `scikit-learn`: Machine Learning model training and evaluation
  - `SpaCy`: NLP text preprocessing
  - `Streamlit`: Web app development
  - `pickle`: Model and pipeline serialization
- **Machine Learning Models**:
  - Logistic Regression (97% accuracy)
  - Multinomial Naive Bayes (92% accuracy)

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/iamathullll/True-News-Validator.git
   cd True-News-Validator
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## **How It Works**
1. **Preprocessing**:
   - Text input is lemmatized, and stopwords/punctuation are removed using SpaCy.
2. **Feature Engineering**:
   - TF-IDF vectorizer converts cleaned text into numerical data.
3. **Classification**:
   - Logistic Regression model predicts the label as "Real" or "Fake."
4. **User Interaction**:
   - Users can input text via a simple Streamlit interface and view classification results instantly.

---

## **Demo**
[Include a link to a video or live app if available.]

---

## **Results**
- Logistic Regression achieved **97% accuracy**, **precision of 94%**, and an **F1-score of 0.92**.
- Multinomial Naive Bayes achieved **92% accuracy**, confirming Logistic Regression as the optimal model.

---

## **Future Enhancements**
- Expand dataset to include multilingual news articles.
- Integrate deep learning models like LSTM or BERT for improved performance.
- Host the application online for broader accessibility.

---

## **Contributing**
Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature name"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## **Contact**
- **Author**: Athul Krishnav V. V.
- **Email**: [your-email@example.com]
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/your-profile)

---
