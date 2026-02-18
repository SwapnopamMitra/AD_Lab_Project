from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import pickle
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)

model = pickle.load(open("model/hybrid_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
accuracy = pickle.load(open("model/accuracy.pkl", "rb"))
model_scores = pickle.load(open("model/model_scores.pkl", "rb"))
feature_names = pickle.load(open("model/feature_names.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", model_scores=model_scores)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {feature: float(request.form.get(feature)) for feature in feature_names}
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    risk = round(probability * 100, 2)
    safe = round(100 - risk, 2)

    if risk < 30:
        category = "Low Risk"
        summary = "Minimal indicators of cardiovascular disease detected."
        risk_class = "risk-low"
    elif risk < 70:
        category = "Moderate Risk"
        summary = "Some clinical indicators suggest potential cardiovascular concern."
        risk_class = "risk-moderate"
    else:
        category = "High Risk"
        summary = "Strong indicators of cardiovascular disease detected. Immediate medical consultation advised."
        risk_class = "risk-high"

    return render_template(
        "index.html",
        prediction_text=category,
        summary=summary,
        risk=risk,
        safe=safe,
        accuracy=round(accuracy * 100, 2),
        model_scores=model_scores,
        risk_class=risk_class,
        form_data=request.form
    )

@app.route("/download_report")
def download_report():
    category = request.args.get("category")
    risk = request.args.get("risk")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Smart Healthcare Diagnostic Report", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("AI-based Heart Disease Risk Assessment", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Predicted Risk Category: {category}", styles["Normal"]))
    elements.append(Paragraph(f"Disease Probability: {risk}%", styles["Normal"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Disclaimer: This AI prediction is not a confirmed medical diagnosis. Please consult a certified medical professional for accurate evaluation.", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Medical_Report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)
