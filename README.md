# Movie Genre Classification

## Overview
This project aims to classify movies based on their genres using traditional machine learning techniques such as Naïve Bayes, Gaussian Naïve Bayes, and Logistic Regression. The classification is based on structured data.

## Features
- Preprocessing of movie dataset (handling missing values, encoding categorical data)
- Exploratory Data Analysis (EDA) to understand genre distribution
- Machine learning models such as Naïve Bayes, Gaussian Naïve Bayes, and Logistic Regression
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
   - Encode categorical features (e.g., genres) using One-Hot Encoding or Label Encoding.
   - Perform feature scaling if necessary.

### 3. Perform Exploratory Data Analysis (EDA)
   - Visualize genre distribution using count plots.
   - Analyze basic statistics of numerical features.

### 4. Split the Dataset
   - Divide data into training and testing sets (e.g., 80%-20%).

### 5. Train a Classification Model
   - Implement various ML models like Naïve Bayes, Gaussian Naïve Bayes, and Logistic Regression.
   - Tune hyperparameters for better performance.

### 6. Evaluate the Model
   - Measure accuracy, precision, recall, and F1-score.
   - Visualize performance using confusion matrix.

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
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
- The trained model achieves reliable accuracy in predicting movie genres using structured data.
- Visualization of genre distribution and model performance metrics.

## Future Improvements
- Implement additional machine learning models like Decision Trees and Random Forest.
- Explore ensemble learning techniques for improved accuracy.
- Deploy the model as a web application using Flask or Streamlit.




---
Happy Coding! 🚀

