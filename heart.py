import streamlit as st
import numpy as np
import pandas as pd
import io
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="Heart Disease Prediction (With Exact Metrics)", layout="centered")
st.title("Heart Disease Prediction System")
st.write("This  web app trains the same models and stacking ensemble as your offline script and shows identical metrics. Fill patient details and download a PDF report.")

# Feature lists (must match your dataset)
NUMERICAL = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
ALL_FEATURES = NUMERICAL + CATEGORICAL

# ---------------- Data load + preprocessor ----------------
@st.cache_resource
def load_and_preprocess_data():
    heart = fetch_ucirepo(id=45)
    X_raw = heart.data.features
    y_raw = heart.data.targets

    # Binarize target exactly like your script
    y = (y_raw['num'] > 0).astype(int)
    y = y.values.ravel()

    # Preprocessor same as offline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, NUMERICAL),
        ('cat', categorical_transformer, CATEGORICAL)
    ], remainder='passthrough')

    return X_raw, y, preprocessor

# ---------------- Train + evaluate (same logic as your offline script) ----------------
@st.cache_resource
def train_and_evaluate_models():
    X, y, preprocessor = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Define models same as offline
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    }

    results = {}
    trained_pipelines = {}

    # Train individual pipelines and compute metrics (Accuracy, AUC-ROC, F1)
    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        # Some classifiers can return predict_proba; for safety, if not available use zeros
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba = np.zeros_like(y_test, dtype=float)
            auc = float('nan')

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'Accuracy': acc,
            'AUC-ROC': auc,
            'F1-Score': f1
        }
        trained_pipelines[name] = pipe

    # Build stacking classifier using pipelines as base estimators (same pattern as offline)
    base_estimators = [
        ('lr', Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models['Logistic Regression'])])),
        ('dt', Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models['Decision Tree'])])),
        ('rf', Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models['Random Forest'])])),
        ('xgb', Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models['XGBoost'])]))
    ]
    final_estimator = LogisticRegression(solver='lbfgs', max_iter=1000)

    stacked_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )

    # fit stacking on training data (preprocessor included inside base estimators)
    stacked_model.fit(X_train, y_train)

    # Evaluate stacked model on the same test split
    y_pred_stack = stacked_model.predict(X_test)
    try:
        y_proba_stack = stacked_model.predict_proba(X_test)[:, 1]
        auc_stack = roc_auc_score(y_test, y_proba_stack)
    except Exception:
        y_proba_stack = np.zeros_like(y_test, dtype=float)
        auc_stack = float('nan')

    acc_stack = accuracy_score(y_test, y_pred_stack)
    f1_stack = f1_score(y_test, y_pred_stack)

    results['Stacked Ensemble'] = {
        'Accuracy': acc_stack,
        'AUC-ROC': auc_stack,
        'F1-Score': f1_stack
    }

    # Return stacked model, preprocessor (for consistency) and the full results dict
    return stacked_model, preprocessor, results

# Train & evaluate (cached)
with st.spinner("Training models (this runs once and is cached)..."):
    stacked_model, preprocessor, results = train_and_evaluate_models()

# Display metrics in a compact table (exact same metrics as offline)
st.subheader("🔍 Model Performance (computed on the same split as offline script)")
metrics_df = pd.DataFrame(results).T[['Accuracy', 'AUC-ROC', 'F1-Score']]
# Convert Accuracy/F1 to percentage for readability; keep AUC as fraction if not NaN
metrics_display = metrics_df.copy()
metrics_display['Accuracy'] = metrics_df['Accuracy'] * 100
metrics_display['F1-Score'] = metrics_df['F1-Score'] * 100
st.dataframe(metrics_display.round(4))

# ---------------- Input form ----------------
st.write("---")
st.header("Patient Input")
with st.form("patient_form"):
    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    trestbps = st.number_input("Resting BP (trestbps)", min_value=50, max_value=300, value=130)
    chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=700, value=200)
    thalach = st.number_input("Max Heart Rate (thalach)", min_value=30, max_value=260, value=150)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.2f")

    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar >120 (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia (thal)", [1.0, 2.0, 3.0])

    submit = st.form_submit_button("Predict & Generate PDF")

# ---------------- Prediction + PDF generation ----------------
if submit:
    # build patient DataFrame in same column order
    patient = pd.DataFrame([[
        age, trestbps, chol, thalach, oldpeak,
        sex, cp, fbs, restecg, exang, slope, ca, thal
    ]], columns=ALL_FEATURES)

    # Predict with stacked model (identical to offline stacking)
    pred = stacked_model.predict(patient)[0]
    try:
        prob = stacked_model.predict_proba(patient)[0][1] * 100
    except Exception:
        # If predict_proba not available, set to 0
        prob = 0.0

    st.write("---")
    st.subheader("📘 Prediction Result")
    if pred == 1:
        st.error(f"🚨 High Risk of Heart Disease: {prob:.2f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease: {prob:.2f}%")

    # Show the exact metrics (matching offline)
    st.write("### 📊 Model Accuracies (Exact values from training/evaluation):")
    for model_name, metrics in results.items():
        acc = metrics['Accuracy'] * 100
        auc = metrics['AUC-ROC']
        f1 = metrics['F1-Score'] * 100
        # show accuracy prominently and AUC/F1 inline
        if np.isnan(auc):
            st.write(f"**{model_name}:** Accuracy = {acc:.2f}%, F1 = {f1:.2f}%")
        else:
            st.write(f"**{model_name}:** Accuracy = {acc:.2f}%, AUC = {auc:.4f}, F1 = {f1:.2f}%")

    st.write("### Patient Input Summary")
    st.table(patient)

    # ---------- PDF creation (beautiful margins, patient name/id, table, models) ----------
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    LEFT = 50
    RIGHT = width - 50
    TOP = height - 50
    LINE_H = 16

    # Title
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawCentredString(width / 2, TOP, "Heart Disease Prediction Report")

    # Patient info
    y = TOP - 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(LEFT, y, "Patient Information:")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(LEFT + 10, y - LINE_H, f"Name   : {patient_name if patient_name else 'N/A'}")
    pdf.drawString(LEFT + 10, y - (2 * LINE_H), f"Patient ID : {patient_id if patient_id else 'N/A'}")

    # Prediction
    y = y - (3 * LINE_H)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(LEFT, y, "Prediction:")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(LEFT + 10, y - LINE_H, f"Result      : {'HIGH RISK' if pred == 1 else 'LOW RISK'}")
    pdf.drawString(LEFT + 10, y - (2 * LINE_H), f"Probability : {prob:.2f}%")

    # Model metrics block
    y = y - (4 * LINE_H)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(LEFT, y, "Model Performance (on test split):")
    pdf.setFont("Helvetica", 10)
    y -= LINE_H
    for model_name, metrics in results.items():
        acc = metrics['Accuracy'] * 100
        auc = metrics['AUC-ROC']
        f1 = metrics['F1-Score'] * 100
        if np.isnan(auc):
            line = f"{model_name}: Accuracy = {acc:.2f}%, F1 = {f1:.2f}%"
        else:
            line = f"{model_name}: Accuracy = {acc:.2f}%, AUC = {auc:.4f}, F1 = {f1:.2f}%"
        pdf.drawString(LEFT + 10, y, line)
        y -= LINE_H

    # Patient values table (two-column)
    y -= (LINE_H * 1.5)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(LEFT, y, "Patient Input Summary:")
    y -= (LINE_H * 1.2)

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(LEFT, y, "Feature")
    pdf.drawString(LEFT + 220, y, "Value")
    y -= (LINE_H * 0.8)
    pdf.line(LEFT, y, RIGHT - 20, y)
    y -= LINE_H

    pdf.setFont("Helvetica", 10)
    for col, val in patient.iloc[0].items():
        pdf.drawString(LEFT, y, str(col))
        pdf.drawString(LEFT + 220, y, str(val))
        y -= LINE_H
        # If we reach bottom margin, create a new page
        if y < 80:
            pdf.showPage()
            y = TOP - 40
            pdf.setFont("Helvetica", 10)

    # Footer notes
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(LEFT, 40, "Note: Metrics above were computed on the same 80/20 train/test split as your offline script.")

    pdf.save()
    buffer.seek(0)

    # Provide download button
    st.download_button(
        label="📄 Download PDF Report",
        data=buffer,
        file_name="heart_disease_report_exact_metrics.pdf",
        mime="application/pdf"
    )
