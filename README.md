# Spam email detector

## Project overview
The goal of this project is to classify spam emails provided by users through a web interface.  
The program analyzes the content of an email and determines whether it is spam or not.  
Additionally, it maintains a history of processed emails for user reference.

---

## Project description
This system is trained using an external dataset containing labeled email samples, where each email is marked as:
- **Spam** (`1`)
- **Not spam** (`0`)

The key stages of the project include:

### 1. Data loading and preparation

#### Data cleaning
The following techniques are used to clean and preprocess the email text:
- **Tokenization**:  
  Splitting the text into smaller units (tokens) while removing unwanted characters such as punctuation.
- **Lemmatization**:  
  Extracting the root form of words based on their context to ensure uniformity in analysis.
- **Stopword removal**:  
  Eliminating commonly used words (e.g., "the," "and") that do not contribute significant meaning.

#### Vectorization
The text is converted into numerical representations using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.  
This approach highlights the most meaningful words in a document relative to the entire dataset.

---

### 2. Model training

#### Data splitting
The dataset is divided into training and testing sets to evaluate model performance.

#### Naive Bayes classifier
The chosen model for spam classification due to its effectiveness in text classification tasks.

#### Model evaluation
- **Accuracy**: Achieved **97%**, demonstrating high reliability.  
- **Confusion matrix**: Provides insights into how well the model distinguishes between spam and non-spam emails.

---

### 3. Web application integration
The web application is built using the **Flask** framework and includes the following features:
1. A user-friendly interface for submitting email content.
2. Backend processing for spam classification.
3. Display of classification results (Spam/Not Spam).
4. A table view of the history of classified emails.

---

## Setup instructions

### Prerequisites
Ensure the following are installed on your system:
- **Python** (version 3.8 or higher)
- **pip** (Python package manager)

### Steps to run

1. **Clone the repository from GitHub**:
   ```bash
   git clone https://github.com/anaastastasiia/spam-email-detector.git
   cd spam-email-detector
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Train the model by running**:
   ```bash
   python model.py
4. **Start the web application**:
   ```bash
   python app.py
5. **Open your browser and navigate to**:
   ```bash
   http://127.0.0.1:5000
