import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from huggingface_hub import hf_hub_download

@st.cache_resource
def download_nltk_resources():
    try:
        # Try downloading the newer punkt_tab resource first
        nltk.download('punkt_tab')
    except:
        # Fallback to the older punkt resource
        nltk.download('punkt')
    
    nltk.download('stopwords')

download_nltk_resources()

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load model and vectorizer with error handling
@st.cache_resource
def load_models():
    try:
        model_path = hf_hub_download(repo_id="imabhishekc/spam-classifier-model", filename="model.pkl")
        vectorizer_path = hf_hub_download(repo_id="imabhishekc/spam-classifier-model", filename="vectorizer.pkl")
        
        model = pickle.load(open(model_path, 'rb'))
        tfidf = pickle.load(open(vectorizer_path, 'rb'))
        
        return model, tfidf
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, tfidf = load_models()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    elif model is None or tfidf is None:
        st.error("Models could not be loaded. Please check your internet connection and try again.")
    else:
        try:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            if result == 1:
                st.header("🚨 Spam")
                st.error("This message appears to be spam.")
            else:
                st.header("✅ Not Spam")
                st.success("This message appears to be legitimate.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")