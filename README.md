
# Fake News Prediction Using Logistic Regression
---

<h4 style="color:#ff4d6d">📰 Project Overview</h4>

A machine learning project to detect fake news using Natural Language Processing (NLP), TF-IDF vectorization, and several classification models. This project includes text preprocessing, word cloud visualizations, model comparisons, and performance evaluation.

---

<h4 style="color:#ff4d6d">🚀 Features</h4>

- 📚 **Text Preprocessing**: Cleaning, tokenization, stemming, stopword removal  
- 🔍 **TF-IDF Vectorization**: Feature extraction from raw text  
- 🤖 **Model Training**: Logistic Regression + Naive Bayes, Random Forest, and SVM  
- 📊 **Performance Metrics**: Accuracy, confusion matrix, and classification report  
- ☁️ **Visualizations**: WordClouds for fake vs real news  
- 🔄 **Model Comparison**: Accuracy of different models on test set  

---

<h4 style="color:#ff4d6d">🧠 Machine Learning Workflow</h4>

- **Step 1:** Import Libraries — `nltk`, `sklearn`, `matplotlib`, `seaborn`, etc.  
- **Step 2:** Download NLTK stopwords for text cleaning  
- **Step 3:** Load Fake & Real news datasets and label them (0 = Fake, 1 = Real)  
- **Step 4:** Merge datasets and explore: shape, missing values, samples  
- **Step 5:** Preprocess text — remove symbols, lowercase, remove stopwords, stemming  
- **Step 6:** TF-IDF Vectorization  
- **Step 7:** Train/test split (70% train, 30% test)  
- **Step 8:** Train Logistic Regression and evaluate accuracy  
- **Step 9:** Predict a random article to test classification  
- **Step 10:** Generate WordClouds for Fake and Real articles  
- **Step 11:** Train & evaluate Naive Bayes, Random Forest, and SVM models  

---

<h4 style="color:#ff4d6d">🛠️ Tech Stack</h4>

| Layer       | Technology                       |
|-------------|----------------------------------|
| Language    | Python 3.12                      |
| Libraries   | NLTK, Scikit-learn, Matplotlib   |
| Vectorizer  | TF-IDF (TfidfVectorizer)         |
| Models      | Logistic Regression, SVM, Naive Bayes, Random Forest |
| Visualization | WordCloud, Seaborn, Matplotlib |
| Dataset     | Fake and True CSV News Datasets  |

---

<h4 style="color:#ff4d6d">📁 Project Structure</h4>

```text
├── Data/
│   ├── Fake.csv                        ← Fake news dataset
│   └── True.csv                        ← Real news dataset
├── Fake News Prediction.ipynb         ← Main Jupyter Notebook
├── outputs/                            ← WordCloud images, optional
└── README.md                           ← Project documentation
