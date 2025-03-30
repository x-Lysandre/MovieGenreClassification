# Movie Genre Classification

## Overview
This project aims to classify movies based on their genres using machine learning and natural language processing (NLP) techniques. By leveraging movie titles and descriptions as input features, the model predicts the appropriate genre(s) for a given movie.

## Features
- Preprocessing of movie dataset (handling missing values, text cleaning, feature extraction)
- Exploratory Data Analysis (EDA) to understand genre distribution
- Machine learning models such as Logistic Regression, SVM, Random Forest, and XGBoost
- Deep learning approaches using LSTMs, CNNs, and BERT for improved text classification
- Model evaluation using accuracy, precision, recall, and F1-score
- Visualization of results

## Dataset
The dataset consists of:
- **Movie Title:** The name of the movie
- **Movie Description:** A short synopsis of the movie plot
- **Genre:** The category or categories assigned to the movie (e.g., Action, Drama, Comedy)

## Steps to Perform
### 1. Load the Data
   - Import dataset using Pandas.
   - Check for missing values and handle them appropriately.

### 2. Preprocess the Data
   - Convert text data into numerical format using TF-IDF or Word Embeddings.
   - Encode categorical features (e.g., genres) using One-Hot Encoding or Multi-Label Binarization.
   - Perform feature scaling if necessary.

### 3. Perform Exploratory Data Analysis (EDA)
   - Visualize genre distribution using count plots.
   - Analyze word frequency in movie descriptions.

### 4. Split the Dataset
   - Divide data into training and testing sets (e.g., 80%-20%).

### 5. Train a Classification Model
   - Implement various ML models like Logistic Regression, SVM, Random Forest, and XGBoost.
   - Experiment with deep learning models like LSTMs and CNNs for better performance.

### 6. Evaluate the Model
   - Measure accuracy, precision, recall, and F1-score.
   - Visualize performance using confusion matrix.

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib nltk tensorflow torch transformers
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-genre-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd movie-genre-classification
   ```
3. Run the script:
   ```bash
   python train_model.py
   ```

## Results
- The trained model achieves high accuracy in predicting movie genres based on text features.
- Visualization of genre distribution and model performance metrics.

## Future Improvements
- Implement ensemble learning techniques.
- Experiment with transformer models like BERT for enhanced classification.
- Deploy the model as a web application using Flask or Streamlit.

## Contributing
Feel free to submit issues or pull requests to improve this project.

## License
This project is licensed under the MIT License.

---
Happy Coding! 🚀


