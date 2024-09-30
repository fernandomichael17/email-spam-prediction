import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download resources untuk NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Fungsi untuk membersihkan teks
def cleaning_text(text):
    # Mengambil punctuation dan stopwords
    punc = list(string.punctuation)
    stop = stopwords.words('english')
    
    # Menggabungkan punctuation dan stopwords
    prob = punc + stop
    
    # Inisialisasi lemmatizer
    lemma = WordNetLemmatizer()
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Menghapus karakter non-alfabet
    word_tokens = [t for t in tokens if t.isalpha()]
    
    # Membersihkan teks dengan lemmatization dan penghapusan stopwords
    clean = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in prob]
    
    return ' '.join(clean)

# Memuat model dan vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit Interface
st.title("Email Spam Detection")

# Menampilkan gambar
st.image("email_spam_image.png", use_column_width=True)

# Membuat input untuk email
email_input = st.text_area("Masukkan teks email di sini:", height=200)

# Tombol untuk memulai prediksi
if st.button("Prediksi"):
    if email_input:
        # Praproses teks menggunakan fungsi cleaning_text
        cleaned_text = cleaning_text(email_input)
        
        # Transformasi teks yang sudah dibersihkan menggunakan TF-IDF Vectorizer
        email_features = vectorizer.transform([cleaned_text])
        
        # Melakukan prediksi (hasil 0 = bukan spam, 1 = spam)
        prediction = model.predict(email_features)[0]
        
        # Menampilkan hasil prediksi
        if prediction == 1:
            st.error("Email ini terdeteksi sebagai SPAM.")
        else:
            st.success("Email ini BUKAN SPAM.")
    else:
        st.warning("Harap masukkan teks email.")
