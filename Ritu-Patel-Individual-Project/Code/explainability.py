# LIME Explainability for Suicide Ideation Detection
from lime.lime_text import LimeTextExplainer
import joblib
import pandas as pd

MODEL_PATH = "baseline_lr_model.pkl"
TFIDF_PATH = "baseline_tfidf.pkl"
DATA_PATH = "Suicide_Detection.csv"

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

df = pd.read_csv(DATA_PATH)
df = df[['text', 'class']].dropna().reset_index(drop=True)

class_names = ["non-suicide", "suicide"]

explainer = LimeTextExplainer(class_names=class_names)

# Pick one sample suicide text from the dataset
idx = 150
text_sample = df['text'][idx]

def predict_proba(texts):
    features = tfidf.transform(texts)
    return model.predict_proba(features)

exp = explainer.explain_instance(
    text_sample,
    predict_proba,
    num_features=10
)

# Save LIME explanation instead of trying to show it in notebook
output_file = "lime_explanation_suicide.html"
exp.save_to_file(output_file)
print(f"LIME explanation saved as: {output_file}")


