Fake News Detection using Machine Learning

This project is a full-stack Machine Learning application designed to identify fabricated news articles. By utilizing **Natural Language Processing (NLP)** and **Logistic Regression**, the system analyzes linguistic patterns to classify news as either **Reliable** or **Unreliable**.

---
1.Features
Real-time Analysis: High-speed prediction using a trained Logistic Regression model.

Interactive UI: A modern, professional dashboard built with Streamlit.

Linguistic Metrics: Displays word count and character count for input text.

Forensic Diagnostic: Simulated scanning progress to enhance user trust.

2. Tech Stack
Frontend: Streamlit

Machine Learning: Scikit-Learn (Logistic Regression)

Natural Language Processing: NLTK (Natural Language Toolkit)

Data Manipulation: Pandas, NumPy

Serialization: Pickle

3. Detailed Explanation
1. Data Preprocessing

Raw text is cleaned to ensure the model focuses only on meaningful data. This involves:

Stemming: Reducing words to their root form (e.g., "running" becomes "run").

Vectorization: Using TfidfVectorizer to convert text into numerical weights based on word importance.

2. The Model

The Logistic Regression model was trained on a labeled dataset. During training, it learned to associate specific weighted word patterns with "Real" or "Fake" labels.

Class 1: Real News

Class 0: Fake News

3. Web Interface

The app.py script bridges the gap between the saved .pkl files and the user. It applies the same preprocessing used during training to the user's input before running the prediction.

4.Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/yourusername/Fake-News-Detection.git](https://github.com/yourusername/Fake-News-Detection.git)
cd Fake-News-Detection
Install dependencies:

Bash
pip install streamlit scikit-learn nltk pandas
Download NLTK Data: The app will automatically attempt to download stopwords via the code, but you can do it manually:

Python
import nltk
nltk.download('stopwords')
Run the Application:

Bash
streamlit run app.py
