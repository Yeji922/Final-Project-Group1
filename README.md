# Final Project — Group 1  
The following project aimed to develop an NLP-based platform to analyze whether text messages sent by individuals can be classified as suicidal or non-suicidal.


# Dataset
The data used for this project is the “Suicide and Depression Detection” dataset from Kaggle. https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data


## 📝 Overview  
This repository contains the complete work produced by **Group 1** for the course's final project.  
It includes:

- The **final group report**
- The **final presentation slides**
- The **project proposal**
- **Individual work** submitted by each team member
- A collection of **code notebooks and scripts**

This structure ensures that all shared and individual contributions are easy to navigate, review, and assess.



## 📂 Repository Structure  

Final-Project-Group1/
```
├── Code/
│  ├── baseline/      - Code implementing classical NLP baseline models  
│  ├── lstm/          - Code for models built using BiLSTM architecture  
│  ├── bert/          - Code for models developed through BERT fine-tuning  
│  ├── app/           - Code that develops and runs the Streamlit application 

|
├── Final-Group-Project-Report/
│  ├── Final_Report.pdf
│
├── Final-Group-Presentation/
│ ├── Final_Presentation.pptx
│
├── Group-Proposal/
│ ├── Group_Proposal_Suicide.pdf
│
├── <firstname>-<lastname>-individual-project/
│ ├── Code/ - All code files developed by individual contributors respectively
| ├── <firstname>-<lastname>-individual-project/
│      ├── <firstname>-<lastname>-final-project.pdf
│
└── README.md
```


---

## 🎯 Project Goal  
The primary goal of this project is to develop an NLP-based system capable of accurately classifying user-generated text as suicidal or non-suicidal. In today’s digital environment, individuals experiencing suicidal ideation often express distress online before taking action. This project aims to leverage these linguistic signals to provide early detection that could support timely intervention.
To achieve this goal, the project is guided by the following objectives:

1. Build a Multi-Model Suicide Ideation Classifier
Develop and compare multiple Natural Language Processing models—including a classical TF-IDF + Logistic Regression baseline, a Bidirectional LSTM with Attention, and a fine-tuned BERT transformer model—to determine which approach provides the strongest performance for identifying suicidal language.
2. Analyze and Understand Linguistic Patterns Associated With Suicidal Ideation
Study the vocabulary, emotional cues, and contextual patterns that differentiate suicidal posts from ordinary conversation using explainability tools such as LIME, Attention visualization, Integrated Gradients, and SHAP. This ensures the system is not only accurate but also transparent and clinically interpretable.
3. Evaluate and Compare Model Performance Using Reliable Metrics
Assess each model through a rigorous experimental framework, incorporating metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. This allows for a valid comparison of model strengths and weaknesses across different levels of complexity.
4. Create a User-Friendly Application for Real-Time Classification
Develop an interactive Streamlit application that enables users to input text and instantly see predictions from all three models, along with confidence scores. The app serves both as a demonstration tool and a potential foundation for future real-world deployment.
---

## 🧠 Methods & Approach  
The general project workflow includes:

1. **Data Acquisition & Cleaning**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature Engineering / Pre-processing**  
4. **Model Development**  
5. **Model Evaluation**  
6. **Visualization of Insights**
7. **Written & Presentation Deliverables**

All computational work is located inside the `Code/` directory.

---

## ⚙️ Setup & Running Code  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Yeji922/Final-Project-Group1.git
cd Final-Project-Group1
```

## Install Dependencies (if applicable)
```
pip install -r requirements.txt
```
## Run Notebooks or Scripts
```
cd Code
```
## Team members:
Anusha Umashankar - anushau@gwmail.gwu.edu

Yeji Kim - yeji.kim@gwmail.gwu.edu

Ritu Patel - ritu.patel@gwu.edu,

Fardin Hafiz - fardin.hafiz@gwu.edu




