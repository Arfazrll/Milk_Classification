# Data Classification & Model Evaluation

## Project Overview

This project demonstrates the application of machine learning algorithms to solve a classification problem. The project focuses on addressing **class imbalance** by using **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure that the models perform fairly and accurately.

### Team Members:
- **Asep Jamaludin**
- **Syahril Arfian Almazril**
- **Ridho Anugrah Mulyadi**
- **Sayyid Faqih**

The project includes key steps such as **data preprocessing**, **model training**, **evaluation**, and handling class imbalance for improving model fairness.

---

## Key Features

### 1. **Data Preprocessing**
   - **Cleaning**: Addressed missing values, outliers, and invalid data.
   - **Feature Engineering**: Created new features to improve model performance.
   - **Normalization & Scaling**: Applied **StandardScaler** to scale features to ensure models converge more efficiently.

### 2. **Machine Learning Models**
   We applied and evaluated multiple machine learning models:
   - **Support Vector Machine (SVM)**: Effective for high-dimensional data.
   - **K-Nearest Neighbors (KNN)**: A distance-based classifier known for its simplicity and efficiency.
   - **Naive Bayes**: A probabilistic classifier based on Bayes’ Theorem, ideal for high-dimensional data.

### 3. **Class Imbalance Handling**
   - **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to generate synthetic data for the minority class to mitigate class imbalance, improving model performance.

### 4. **Model Evaluation**
   - **Evaluation Metrics**: Used **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix** to evaluate model performance.
   - **Cross-validation**: Performed k-fold cross-validation to ensure the models generalize well and are not overfitted.

---

## Technologies Used

- **Python 3.x**: Programming language for data manipulation, modeling, and evaluation.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Numerical operations.
- **Scikit-learn**: Machine learning library for model training, evaluation, and metrics.
- **Imbalanced-learn (SMOTE)**: For addressing class imbalance.
- **Matplotlib & Seaborn**: For data visualization.
- **Jupyter Notebook**: For interactive development and documentation.

---

## Results

The following are the performance results for each model:

### 1. **Support Vector Machine (SVM)**
   - **Accuracy**: 85.6%
   - **Precision**: 84.3%
   - **Recall**: 86.2%
   - **F1 Score**: 85.2%

### 2. **K-Nearest Neighbors (KNN)**
   - **Accuracy**: 82.1%
   - **Precision**: 80.2%
   - **Recall**: 83.1%
   - **F1 Score**: 81.6%

### 3. **Naive Bayes**
   - **Accuracy**: 79.4%
   - **Precision**: 77.8%
   - **Recall**: 80.3%
   - **F1 Score**: 78.9%

**Key Insight**: The **SVM** model performed the best, showing the highest accuracy and F1 score, though **KNN** also produced strong results.

---

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd repository-name
   ```

3. **Install Required Dependencies**:
   Install the necessary libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**:
   Launch Jupyter Notebook and open the `project_analysis.ipynb` to explore and run the project:
   ```bash
   jupyter notebook
   ```

---

## Key Insights & Takeaways

- **SMOTE**: Addressing class imbalance with **SMOTE** was crucial in improving the fairness and performance of the models, especially in the context of imbalanced datasets.
  
- **Feature Scaling**: Properly scaling features significantly improved the performance of models like **SVM** and **KNN**, which are sensitive to the scale of the data.

- **Model Evaluation**: Using a variety of metrics like **precision**, **recall**, and **F1 score** provided a more comprehensive understanding of model performance, especially in imbalanced scenarios.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contributing

We welcome contributions! If you have any suggestions or improvements, feel free to fork the repository, open an issue, or submit a pull request.

---

## Acknowledgements

Special thanks to the following resources that contributed to the success of this project:

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

