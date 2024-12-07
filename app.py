from flask import Flask, request, jsonify, render_template
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def tokenize_message(message):
    return [token for token in word_tokenize(message) if token.isalpha()]

lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def preprocess_email(email):
    tokens = tokenize_message(email)
    lemmatized_tokens = lemmatize_tokens(tokens)
    filtered_tokens = remove_stopwords(lemmatized_tokens)
    return ' '.join(filtered_tokens)

# aplikacja
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.form.get('email')
    if not data:
        return jsonify({'error': 'No email content provided!'}), 400

    processed_email = preprocess_email(data)
    email_vector = vectorizer.transform([processed_email])
    prediction = model.predict(email_vector)[0]
    prediction_label = 'Spam' if prediction == 1 else 'Not Spam'

    return jsonify({'email': data, 'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
