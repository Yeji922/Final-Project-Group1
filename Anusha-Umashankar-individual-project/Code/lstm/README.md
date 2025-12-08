# LSTM Attention-Based Text Classification
### *With Evaluation Metrics, Explainability (XAI), and Model Interpretation*

This folder contains a complete pipeline for training, evaluating, and explaining a **BiLSTM + Attention** model for text classification.  
The project is designed for academic NLP coursework, with a strong focus on **model explainability**, **interpretability**, and **evaluation metrics**.

## ✨ Project Highlights

- **Bidirectional LSTM with Attention**  
  Learns contextual representations of input text with token-level attention weights.

- **Full NLP Pipeline**  
  Includes preprocessing → training → evaluation → prediction → explainability.

- **Advanced Explainability (XAI) Module**  
  Uses:
  - Attention heatmaps  
  - Highlighted text visualization  
  - Attention weight distribution  
  - Integrated Gradients (Captum)  
  - SHAP values  

- **Training + Prediction CLI**  
  Easily train or run inference using `main.py`.

## 📂 Folder Structure

```
lstm/
│
└── Utils/
       ├── __init__.py     
       ├── vocab.pkl
       ├── advanced_xai.py
       ├── dataloader.py  
       ├── evaluation.py
       ├── explainability.ipynb
       ├── explainability.py
       ├── model.py
       ├── predict.py
       ├── train.py
├── main.py                              
├── Suicide_detection.csv       
├── README.md
└── saved_model/
       ├── lstm_attention_model.pt  
       ├── log_file.txt  
       ├── vocab.pkl                   
       ├── label_encoder.pkl           
       ├── attention_plot.png          
       ├── confusion_matrix.png        
       └── ig_sample.png / lrp_sample.png / shap outputs
```

## How to Use This Folder

### 1. Train the Model

```
python main.py --opt train --path "Suicide_detection.csv"
```

Dataset format:

| Column | Description |
|--------|-------------|
| text   | input text sentence |
| class  | label for classification |

### 2. Evaluation

Automatically runs after training:

- Accuracy  
- Precision  
- Recall  
- Macro-F1  
- Confusion matrix  
- Classification report  

Saved in `saved_model/`.

### 3. Predict on New Text

```
python main.py --opt predict --path saved_model/lstm_attention_model.pt,saved_model/vocab.pkl,saved_model/label_encoder.pkl
```

Outputs:

- Predicted class  
- Attention heatmap  
- Highlighted text  
- Distribution plot  
- Attention token mapping  

## Explainability / XAI Tools

### Attention-Based:
- `plot_attention()`  
- `highlight_text()`  
- `plot_attention_distribution()`  
- `get_attention_mapping()`

### Advanced XAI:
- Integrated Gradients  
- SHAP  

Run via `Explainability_Analysis.ipynb`.

## Notebook

The notebook shows:

- Attention visualizations  
- IG, SHAP  
- Saved plots for reports  

##  Model Architecture

```
Embedding → BiLSTM → Attention → Context Vector → Fully Connected → Softmax
```

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- Macro-F1  
- Confusion Matrix  
- Classification Report  

## Requirements

```
pip install torch numpy pandas scikit-learn matplotlib seaborn captum shap
```
