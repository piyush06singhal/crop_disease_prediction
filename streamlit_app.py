# streamlit_app.py - Streamlit Demo for Crop Disease Prediction
"""
Simple Streamlit demo application for Crop Disease Prediction System.
This is a lightweight alternative for quick demos and testing.
"""

import streamlit as st
import requests
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Crop Disease Prediction",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #4caf50;
    }
    .result-card {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 25px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🌱 Crop Disease Prediction System</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### AI-Powered Plant Disease Detection
    Upload a leaf image to detect crop diseases using advanced machine learning.
    Supports tomato, potato, corn, and other common crops.
    """)

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease detection"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded leaf image", use_column_width=True)

        with col2:
            st.subheader("🔍 Analysis Results")

            # Show loading spinner
            with st.spinner("Analyzing image..."):
                try:
                    # For demo purposes, simulate API call
                    # In real implementation, call your Flask API
                    result = simulate_prediction(uploaded_file)

                    # Display results
                    display_results(result)

                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
                    st.info("💡 Make sure the image shows a clear leaf and try again.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🧠 AI Technology**
        - MobileNetV2 deep learning model
        - 38+ disease classes
        - Real-time inference
        """)

    with col2:
        st.markdown("""
        **🌾 Supported Crops**
        - Tomato
        - Potato
        - Corn
        - Pepper
        - Apple
        - Grape
        """)

    with col3:
        st.markdown("""
        **📊 Features**
        - Disease detection
        - Confidence scoring
        - Treatment suggestions
        - Progress tracking
        """)

def simulate_prediction(uploaded_file):
    """Simulate prediction for demo purposes"""
    # In a real implementation, this would call your Flask API
    # For now, return mock results

    diseases = [
        "Tomato Healthy",
        "Tomato Leaf Blight",
        "Tomato Bacterial Spot",
        "Potato Healthy",
        "Potato Late Blight",
        "Corn Healthy",
        "Corn Common Rust"
    ]

    import random
    disease = random.choice(diseases)
    confidence = random.uniform(0.75, 0.98)

    return {
        "disease": disease,
        "confidence": confidence,
        "crop_type": disease.split()[0],
        "treatment": get_treatment_suggestion(disease),
        "prevention": get_prevention_tips(disease)
    }

def display_results(result):
    """Display prediction results"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # Disease prediction
    st.success(f"**Disease Detected:** {result['disease']}")

    # Confidence score
    confidence_percent = result['confidence'] * 100
    st.metric("Confidence Score", f"{confidence_percent:.1f}%")

    # Confidence bar
    if confidence_percent >= 80:
        color = "🟢"
    elif confidence_percent >= 60:
        color = "🟡"
    else:
        color = "🔴"

    st.progress(confidence_percent / 100)
    st.caption(f"{color} High confidence" if confidence_percent >= 80 else f"{color} Moderate confidence")

    st.markdown('</div>', unsafe_allow_html=True)

    # Treatment suggestions
    if result.get('treatment'):
        st.subheader("💊 Treatment Recommendations")
        st.info(result['treatment'])

    # Prevention tips
    if result.get('prevention'):
        st.subheader("🛡️ Prevention Tips")
        st.info(result['prevention'])

def get_treatment_suggestion(disease):
    """Get treatment suggestions based on disease"""
    treatments = {
        "Tomato Leaf Blight": "Apply copper-based fungicide. Remove affected leaves. Improve air circulation.",
        "Tomato Bacterial Spot": "Use copper fungicide. Avoid overhead watering. Rotate crops annually.",
        "Potato Late Blight": "Apply fungicide immediately. Remove infected plants. Store potatoes properly.",
        "Corn Common Rust": "Apply fungicide if needed. Plant resistant varieties. Improve field drainage.",
    }
    return treatments.get(disease, "Consult local agricultural extension service for specific treatment recommendations.")

def get_prevention_tips(disease):
    """Get prevention tips based on disease"""
    tips = {
        "Tomato Leaf Blight": "Water at soil level. Space plants for air circulation. Mulch to prevent soil splash.",
        "Tomato Bacterial Spot": "Use disease-resistant varieties. Clean tools between plants. Avoid wet foliage.",
        "Potato Late Blight": "Plant certified seed potatoes. Avoid overhead irrigation. Destroy volunteer plants.",
        "Corn Common Rust": "Plant early. Use resistant hybrids. Avoid excessive nitrogen fertilization.",
    }
    return tips.get(disease, "Practice crop rotation, use certified seeds, and maintain proper plant spacing.")

if __name__ == "__main__":
    main()