import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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

data = pd.read_csv('spam_NLP.csv')

data_cleaned = data.dropna().drop_duplicates(subset=["MESSAGE"])
print(f"Liczba rekordów po oczyszczeniu: {len(data_cleaned)}")

data_cleaned["PROCESSED_MESSAGE"] = data_cleaned["MESSAGE"].apply(preprocess_email)

vectorizer = TfidfVectorizer(min_df=3, max_features=2000)
X = vectorizer.fit_transform(data_cleaned["PROCESSED_MESSAGE"])
y = data_cleaned["CATEGORY"]

print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.2f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', values_format='d')

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model i wektoryzator zostały zapisane do odpowiednich plików.")
