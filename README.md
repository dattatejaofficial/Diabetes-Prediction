# Diabetes Prediction using Machine Learning

This project aims to predict the likelihood of a person having diabetes based on various health-related attributes using machine learning techniques. Early detection of diabetes is crucial for timely intervention and management.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The project utilizes the Pima Indians Diabetes Database, which includes features such as the number of pregnancies, glucose concentration, blood pressure, skinfold thickness, insulin level, BMI, diabetes pedigree function, and age. Various machine learning algorithms, including Logistic Regression, Decision Tree, and Random Forest, are applied to predict the presence of diabetes.

**Key Steps in the Analysis:**

1. **Data Preprocessing:**
   - Load and clean the dataset.
   - Handle missing values and outliers.
   - Split the data into training and testing sets.

2. **Model Building:**
   - Implement different machine learning models.
   - Train the models using the training dataset.

3. **Model Evaluation:**
   - Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Compare the performance of different models.

4. **Prediction:**
   - Use the trained model to predict diabetes onset in new data.

## Project Structure

The repository contains the following files:

- `diabetes.csv`: The dataset used for training and testing the models.
- `diabetes_prediction.ipynb`: A Jupyter Notebook that includes code for data preprocessing, model building, training, and evaluation.

## Setup Instructions

To set up and run the project locally, follow these steps:

1. **Clone the Repository:**
   Use the following command to clone the repository to your local machine:

   ```bash
   git clone https://github.com/dattatejaofficial/Diabetes-Prediction.git
   ```

2. **Navigate to the Project Directory:**
   Move into the project directory:

   ```bash
   cd Diabetes-Prediction
   ```

3. **Create a Virtual Environment (optional but recommended):**
   Set up a virtual environment to manage project dependencies:

   ```bash
   python3 -m venv env
   ```

   Activate the virtual environment:

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. **Install Dependencies:**
   Install the required Python packages using pip:

   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```

## Usage

To run the analysis:

1. **Ensure the Virtual Environment is Activated:**
   Make sure your virtual environment is active (refer to the setup instructions above).

2. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open `diabetes_prediction.ipynb` in the Jupyter interface and execute the cells sequentially to perform the analysis.

## Dependencies

The project requires the following Python packages:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

These dependencies are essential for data manipulation, model building, and visualization.
