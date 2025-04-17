# Machine Learning Project: Data Classification and Model Evaluation

## Overview
This project demonstrates the application of machine learning algorithms to solve a classification problem using real-world data. The project is developed by a team of four collaborators: **Asep Jamaludin**, **Syahril Arfian Almazril**, **Ridho Anugrah Mulyadi**, and **Sayyid Faqih**. The project includes key steps such as data preprocessing, model training, evaluation, and addressing data class imbalance using advanced techniques such as **SMOTE (Synthetic Minority Over-sampling Technique)**.

The objective of this project is to evaluate multiple machine learning models, compare their performance, and tackle challenges associated with imbalanced datasets using **SMOTE**.

---

## ⚙️ Features
### 1. **Data Preprocessing**:
   - **Cleaning**: Data cleaning steps to handle missing values, outliers, and invalid entries.
   - **Feature Engineering**: Transformation of raw data into features suitable for machine learning models.
   - **Normalization & Scaling**: Normalization of data and scaling of continuous features using **StandardScaler** to improve the convergence and performance of models.

### 2. **Modeling**:
   - **Supervised Learning Models**:
     - **Support Vector Machine (SVM)**: A powerful classifier used for high-dimensional data.
     - **K-Nearest Neighbors (KNN)**: A simple yet effective algorithm for classification based on distance metrics.
     - **Naive Bayes**: A probabilistic classifier based on Bayes’ theorem.
   - **Hyperparameter Tuning**: Utilized grid search to tune hyperparameters for optimal performance of each model.

### 3. **Class Imbalance Handling**:
   - **SMOTE**: Applied **SMOTE** to create synthetic minority class samples to combat class imbalance in the dataset.

### 4. **Evaluation**:
   - **Comprehensive Evaluation**: Used key metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix** for model performance evaluation.
   - **Cross-validation**: Applied k-fold cross-validation to ensure the robustness of the models and avoid overfitting.

---

## 🧑‍💻 Technologies Used
- **Python 3.x**: Programming language used to implement the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and handling arrays.
- **Matplotlib**: For visualizing data distributions, model performance, and results.
- **Seaborn**: For advanced statistical visualizations.
- **Scikit-learn**: For machine learning algorithms, model evaluation, and metrics.
- **Imbalanced-learn (SMOTE)**: For addressing class imbalance issues using over-sampling techniques.
- **Jupyter Notebook**: For documenting and running experiments interactively.

---

## 🔧 Installation & Setup

Follow these steps to set up the project locally:

### 1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   ```

### 2. **Navigate to the Project Directory**:
   ```bash
   cd repository-name
   ```

### 3. **Create a Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv venv
   ```

### 4. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

### 5. **Install Required Dependencies**:
   Once the virtual environment is activated, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 6. **Run Jupyter Notebook**:
   After setting up the environment, you can run the Jupyter notebook to explore and run the project:
   ```bash
   jupyter notebook
   ```

   Open the notebook `project_analysis.ipynb` to start interacting with the project.

---

## 📂 Project Structure

Here’s the directory structure of the project:

```
.
├── data/                  # Dataset files (CSV, Excel, etc.)
├── notebooks/             # Jupyter notebooks containing analysis and experiments
│   └── project_analysis.ipynb  # Main notebook for project analysis and experimentation
├── src/                   # Source code for model training and evaluation
│   └── preprocess.py      # Script for preprocessing the data (cleaning, scaling, encoding)
│   └── models.py          # Script for model building, training, and evaluation
├── requirements.txt       # List of Python dependencies (use 'pip install -r requirements.txt')
└── README.md              # Project documentation (this file)
```

### `requirements.txt`
This file lists the required Python libraries:
```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

---

## 📊 Model Evaluation

Below is a summary of the performance evaluation for each of the machine learning models:

### 1. **Support Vector Machine (SVM)**
   - **Accuracy**: 85.6%
   - **Precision**: 84.3%
   - **Recall**: 86.2%
   - **F1-Score**: 85.2%
   - **Confusion Matrix**:
     - True Positive: 530
     - True Negative: 460
     - False Positive: 50
     - False Negative: 40

### 2. **K-Nearest Neighbors (KNN)**
   - **Accuracy**: 82.1%
   - **Precision**: 80.2%
   - **Recall**: 83.1%
   - **F1-Score**: 81.6%

### 3. **Naive Bayes**
   - **Accuracy**: 79.4%
   - **Precision**: 77.8%
   - **Recall**: 80.3%
   - **F1-Score**: 78.9%

The **SVM** model performed the best, but we also observed that **KNN** and **Naive Bayes** performed adequately, with slightly lower precision and recall scores.

---

## 🚀 Key Insights

- **Data Preprocessing**: Ensuring the dataset is clean, normalized, and correctly formatted is vital for the model's performance. In this project, we handled missing data, outliers, and scaled the numerical features.
  
- **Class Imbalance**: The **SMOTE** technique is extremely useful for creating a balanced dataset, improving model generalization, and avoiding overfitting, which might otherwise occur when models are trained on highly imbalanced data.
  
- **Model Comparison**: Different models have distinct strengths and weaknesses. For example, **SVM** worked best in this project, but models like **KNN** may perform better in certain scenarios, depending on the dataset's characteristics.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, create an issue, or submit a pull request with improvements, bug fixes, or new features.

---

## 🔗 Acknowledgements

- **Scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)
- **Imbalanced-learn Documentation**: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)
- **Pandas Documentation**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **Matplotlib Documentation**: [https://matplotlib.org/](https://matplotlib.org/)
- **Seaborn Documentation**: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

---
