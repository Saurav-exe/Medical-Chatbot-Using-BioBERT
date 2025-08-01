import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
from huggingface_hub import hf_hub_download
import torch
import mysql.connector
import pickle
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import PyPDF2
import io
from typing import List, Tuple

# Download stopwords
nltk.download('stopwords')

# Hugging Face repo
HF_MODEL_REPO = "Saurav-exe/Hackathon"

# ------------------------- File Processing Functions -------------------------
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_txt(txt_file) -> str:
    """Extract text from uploaded text file"""
    try:
        # Read the file content
        content = txt_file.read()
        # Decode if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content.strip()
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return ""

def process_uploaded_files(uploaded_files) -> str:
    """Process all uploaded files and return combined text"""
    combined_text = ""
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                st.info(f"📄 Processing PDF: {uploaded_file.name}")
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    combined_text += f"\n--- Content from {uploaded_file.name} ---\n{pdf_text}\n"
                    
            elif file_extension in ['txt', 'text']:
                st.info(f"📝 Processing text file: {uploaded_file.name}")
                txt_text = extract_text_from_txt(uploaded_file)
                if txt_text:
                    combined_text += f"\n--- Content from {uploaded_file.name} ---\n{txt_text}\n"
                    
            else:
                st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
    
    return combined_text.strip()

# ------------------------- DB Connection -------------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Password",
        database="Medical_Chatbot"
    )

def get_patient_history(name):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = "SELECT prediction FROM records WHERE name = %s ORDER BY timestamp DESC"
        cursor.execute(query, (name,))
        results = cursor.fetchall()
        history = ". ".join([row[0] for row in results]) if results else ""
        return history
    except Exception as e:
        st.error(f"Database Error: {e}")
        return ""
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def store_prediction(name, prediction):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = "INSERT INTO records (name, prediction, timestamp) VALUES (%s, %s, %s)"
        values = (name, prediction, datetime.now())
        cursor.execute(query, values)
        conn.commit()
    except Exception as e:
        st.error(f"Failed to save prediction: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# ------------------------- Model Loading -------------------------
@st.cache_resource
def load_model():
    try:
        with st.spinner("🔄 Loading model from Hugging Face..."):
            model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO)
            tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO)

            label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.pkl")
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# ------------------------- Text Preprocessing -------------------------
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def enhance_prompt_with_history_and_files(note, file_content=""):
    """Enhanced function to include file content in the prompt"""
    words = note.strip().split()
    if len(words) < 2:
        # If note is too short, use the entire note and file content
        combined_text = f"{note} {file_content}".strip()
        return combined_text, "Unknown Patient"

    name = " ".join(words[:2])
    history = get_patient_history(name)
    current_note = " ".join(words[2:])
    
    # Combine history, current note, and file content
    parts = []
    if history:
        parts.append(f"Patient History: {history}")
    if current_note:
        parts.append(f"Current Note: {current_note}")
    if file_content:
        parts.append(f"Additional Information: {file_content}")
    
    combined = ". ".join(parts) if parts else current_note
    return combined, name

# ------------------------- Prediction Logic -------------------------
disease_data = {
    "Peptic Ulcer Disease": {
        "description": "A sore that develops on the lining of the esophagus, stomach, or small intestine.",
        "medicines": ["Omeprazole", "Pantoprazole", "Ranitidine", "Esomeprazole", "Amoxicillin"],
        "specialists": ["Gastroenterologist", "General Physician", "Internal Medicine Specialist"]
    },
    "Type 2 Diabetes Mellitus": {
        "description": "A chronic condition that affects the way the body processes blood sugar (glucose).",
        "medicines": ["Metformin", "Glipizide", "Insulin", "Sitagliptin", "Canagliflozin"],
        "specialists": ["Endocrinologist", "Diabetologist", "Nutritionist"]
    },
    "Acute Myocardial Infarction": {
        "description": "A medical emergency where the blood flow to the heart is blocked.",
        "medicines": ["Aspirin", "Clopidogrel", "Statins", "Beta Blockers", "ACE Inhibitors"],
        "specialists": ["Cardiologist", "Emergency Medicine Specialist"]
    },
    "Chronic Obstructive Pulmonary Disease": {
        "description": "A group of lung diseases that block airflow and make breathing difficult.",
        "medicines": ["Tiotropium", "Albuterol", "Ipratropium", "Fluticasone", "Salmeterol"],
        "specialists": ["Pulmonologist", "General Physician", "Respiratory Therapist"]
    },
    "Cerebrovascular Accident (Stroke)": {
        "description": "A condition caused by the interruption of blood flow to the brain.",
        "medicines": ["Alteplase", "Aspirin", "Clopidogrel", "Warfarin", "Atorvastatin"],
        "specialists": ["Neurologist", "Rehabilitation Specialist", "Neurosurgeon"]
    },
    "Deep Vein Thrombosis": {
        "description": "A blood clot forms in a deep vein, usually in the legs.",
        "medicines": ["Warfarin", "Heparin", "Apixaban", "Dabigatran", "Rivaroxaban"],
        "specialists": ["Hematologist", "Vascular Surgeon", "Cardiologist"]
    },
    "Chronic Kidney Disease": {
        "description": "The gradual loss of kidney function over time.",
        "medicines": ["Erythropoietin", "Phosphate Binders", "ACE Inhibitors", "Diuretics", "Calcitriol"],
        "specialists": ["Nephrologist", "Dietitian", "Internal Medicine Specialist"]
    },
    "Community-Acquired Pneumonia": {
        "description": "A lung infection acquired outside of a hospital setting.",
        "medicines": ["Amoxicillin", "Azithromycin", "Clarithromycin", "Ceftriaxone", "Levofloxacin"],
        "specialists": ["Pulmonologist", "Infectious Disease Specialist", "General Physician"]
    },
    "Septic Shock": {
        "description": "A severe infection leading to dangerously low blood pressure.",
        "medicines": ["Norepinephrine", "Vancomycin", "Meropenem", "Hydrocortisone", "Dopamine"],
        "specialists": ["Intensivist", "Infectious Disease Specialist", "Emergency Medicine Specialist"]
    },
    "Rheumatoid Arthritis": {
        "description": "An autoimmune disorder causing inflammation in joints.",
        "medicines": ["Methotrexate", "Sulfasalazine", "Hydroxychloroquine", "Adalimumab", "Etanercept"],
        "specialists": ["Rheumatologist", "Orthopedic Specialist", "Physical Therapist"]
    },
    "Congestive Heart Failure": {
        "description": "A chronic condition where the heart doesn't pump blood effectively.",
        "medicines": ["ACE Inhibitors", "Beta Blockers", "Diuretics", "Spironolactone", "Digoxin"],
        "specialists": ["Cardiologist", "General Physician", "Cardiac Surgeon"]
    },
    "Pulmonary Embolism": {
        "description": "A blockage in one of the pulmonary arteries in the lungs.",
        "medicines": ["Heparin", "Warfarin", "Alteplase", "Rivaroxaban", "Dabigatran"],
        "specialists": ["Pulmonologist", "Hematologist", "Emergency Medicine Specialist"]
    },
    "Sepsis": {
        "description": "A life-threatening organ dysfunction caused by a dysregulated immune response to infection.",
        "medicines": ["Vancomycin", "Meropenem", "Piperacillin-Tazobactam", "Cefepime", "Dopamine"],
        "specialists": ["Infectious Disease Specialist", "Intensivist", "Emergency Medicine Specialist"]
    },
    "Liver Cirrhosis": {
        "description": "A late-stage liver disease caused by liver scarring and damage.",
        "medicines": ["Spironolactone", "Furosemide", "Lactulose", "Nadolol", "Rifaximin"],
        "specialists": ["Hepatologist", "Gastroenterologist", "Nutritionist"]
    },
    "Acute Renal Failure": {
        "description": "A sudden loss of kidney function.",
        "medicines": ["Diuretics", "Dopamine", "Calcium Gluconate", "Sodium Bicarbonate", "Epoetin"],
        "specialists": ["Nephrologist", "Critical Care Specialist", "Internal Medicine Specialist"]
    },
    "Urinary Tract Infection": {
        "description": "An infection in any part of the urinary system.",
        "medicines": ["Nitrofurantoin", "Ciprofloxacin", "Amoxicillin-Clavulanate", "Trimethoprim-Sulfamethoxazole", "Cephalexin"],
        "specialists": ["Urologist", "General Physician", "Infectious Disease Specialist"]
    },
    "Hypertension": {
        "description": "A condition in which the force of the blood against the artery walls is too high.",
        "medicines": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide", "Metoprolol"],
        "specialists": ["Cardiologist", "General Physician", "Nephrologist"]
    },
    "Asthma": {
        "description": "A condition in which the airways narrow and swell, causing difficulty in breathing.",
        "medicines": ["Albuterol", "Fluticasone", "Montelukast", "Budesonide", "Salmeterol"],
        "specialists": ["Pulmonologist", "Allergist", "General Physician"]
    },
    "Gastroesophageal Reflux Disease": {
        "description": "A digestive disorder where stomach acid irritates the esophagus.",
        "medicines": ["Omeprazole", "Esomeprazole", "Ranitidine", "Lansoprazole", "Pantoprazole"],
        "specialists": ["Gastroenterologist", "General Physician", "Dietitian"]
    }
}

def predict_disease(text, model, tokenizer, label_encoder):
    if not model or not tokenizer or not label_encoder:
        return "Model not loaded properly"

    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    return label_encoder.inverse_transform([predicted_label])[0]

def get_disease_details(disease_name):
    return disease_data.get(disease_name, {
        "description": "No details available.",
        "medicines": [],
        "specialists": []
    })

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Smart Diagnosis Assistant", layout="centered")

st.title("🧠 Enhanced Disease Predictor")
st.markdown("Enter the patient's note and optionally upload medical documents (PDF/TXT files). The system will use **patient history** and **uploaded files** to enhance the prediction.")

# Load model
model, tokenizer, label_encoder = load_model()

# File upload section
st.subheader("📁 Upload Medical Documents (Optional)")
uploaded_files = st.file_uploader(
    "Choose files to upload", 
    type=['pdf', 'txt', 'text'],
    accept_multiple_files=True,
    help="Upload PDF or text files containing medical reports, lab results, or additional patient information."
)

# Display uploaded files
if uploaded_files:
    st.write("**Uploaded Files:**")
    for file in uploaded_files:
        st.write(f"- {file.name} ({file.type})")

# Text input section
st.subheader("📝 Patient Note")
user_note = st.text_area(
    "Enter patient note", 
    height=150,
    placeholder="Enter patient name and medical notes here..."
)

# Process files and show preview
file_content = ""
if uploaded_files:
    with st.expander("🔍 Preview Uploaded Content", expanded=False):
        file_content = process_uploaded_files(uploaded_files)
        if file_content:
            st.text_area("Combined file content:", file_content[:1000] + "..." if len(file_content) > 1000 else file_content, height=200, disabled=True)
        else:
            st.warning("No content extracted from uploaded files.")

# Prediction section
if st.button("🩺 Predict Disease", type="primary"):
    if user_note:
        # Enhanced prompt with file content
        enhanced_note, patient_name = enhance_prompt_with_history_and_files(user_note, file_content)
        
        # Show what's being processed
        with st.expander("📋 Processing Details", expanded=False):
            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Enhanced Input Length:** {len(enhanced_note)} characters")
            if file_content:
                st.write(f"**File Content Added:** {len(file_content)} characters from {len(uploaded_files)} file(s)")
        
        # Make prediction
        with st.spinner("🔄 Analyzing patient data..."):
            prediction = predict_disease(enhanced_note, model, tokenizer, label_encoder)
        
        # Display results
        st.success(f"🩺 **Predicted Disease:** {prediction}")
        
        # Get and display disease details
        details = get_disease_details(prediction)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📖 Description")
            st.write(details['description'])
            
            st.markdown("### 💊 Common Medicines")
            for medicine in details['medicines']:
                st.write(f"• {medicine}")
        
        with col2:
            st.markdown("### 👨‍⚕️ Recommended Specialists")
            for specialist in details['specialists']:
                st.write(f"• {specialist}")
        
        # Store prediction in database
        store_prediction(patient_name, prediction)
        st.info(f"✅ Prediction saved for patient: {patient_name}")
        
    else:
        st.warning("⚠️ Please enter a patient note to continue.")

# Add instructions
with st.sidebar:
    st.header("📚 How to Use")
    st.markdown("""
    1. **Upload Files (Optional):** Upload PDF or text files containing medical reports, lab results, or patient information.
    
    2. **Enter Patient Note:** Start with patient name followed by medical observations.
    
    3. **Click Predict:** The system will analyze all available information including:
       - Patient's historical diagnoses
       - Current medical note
       - Uploaded document content
    
    4. **View Results:** Get disease prediction with medicines and specialist recommendations.
    """)
    
    st.header("📋 Supported Files")
    st.markdown("""
    - **PDF files:** Medical reports, lab results
    - **Text files:** Clinical notes, patient summaries
    """)
    
    st.header("⚠️ Note")
    st.markdown("""
    This tool is for educational purposes only. 
    Always consult with healthcare professionals for medical decisions.
    """)