import gradio as gr
import joblib

model = joblib.load('spam_svm.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def classify_email(text):
    if not text.strip():
        return "Please enter an email message."
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)
    result = "Spam" if pred == 1 else "Not Spam (Ham)"
    return f"**Prediction**: {result}\n**Confidence**: {confidence:.2%}"

gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(
        placeholder="Hi, claim your $1M prize now! Click here...",
        label="Paste Email Text"
    ),
    outputs=gr.Markdown(),
    title="📧 SVM Spam Detector",
    description="Trained on 5,572 real SMS messages using Support Vector Machine (SVM)",
    examples=[
        ["Free entry in 2 a weekly comp!"],
        ["Hey, are we still meeting tomorrow?"]
    ]
).launch()