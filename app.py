
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
# --- 1. Configuration (These should ideally be consistent with your training) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- 2. Define CLASS_LABELS, disease_solutions, plant_disease_map ---
# These are hardcoded into the app.py for self-containment
CLASS_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

disease_solutions = {
    'Apple___Apple_scab': 'Remove and destroy infected leaves and branches. Apply fungicides during wet periods.',
    'Apple___Black_rot': 'Prune out cankered and diseased wood. Remove mummified fruits. Apply protective fungicides.',
    'Apple___Cedar_apple_rust': 'Remove cedar trees from the vicinity (if possible). Apply fungicides to protect apple trees during spring.',
    'Apple___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Blueberry___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Cherry_(including_sour)___Powdery_mildew': 'Prune out infected shoots. Apply fungicides at the first sign of disease and continue as needed.',
    'Cherry_(including_sour)___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant hybrids. Rotate crops. Plow under infected residue. Apply fungicides if disease is severe.',
    'Corn_(maize)___Common_rust_': 'Use resistant hybrids. Apply fungicides if disease pressure is high, especially for sweet corn.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant hybrids. Rotate crops. Plow under infected residue. Apply fungicides if disease is severe.',
    'Corn_(maize)___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Grape___Black_rot': 'Sanitize vineyard by removing mummified berries and infected canes. Apply fungicides during critical growth stages.',
    'Grape___Esca_(Black_Measles)': 'Prune out infected wood. Apply pruning wound protectants. Maintain vine vigor.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Improve air circulation. Apply fungicides as recommended during the growing season.',
    'Grape___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure. Remove infected trees and control Asian citrus psyllid vectors with insecticides.',
    'Peach___Bacterial_spot': 'Use resistant varieties. Apply copper sprays during dormancy and early spring. Prune to improve air circulation.',
    'Peach___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Pepper,_bell___Bacterial_spot': 'Use disease-free seeds/transplants. Apply copper-based bactericides. Rotate crops.',
    'Pepper,_bell___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Potato___Early_blight': 'Use resistant varieties. Rotate crops. Apply fungicides preventatively and at first signs of disease.',
    'Potato___Late_blight': 'Use resistant varieties. Apply fungicides preventatively and at first signs of disease. Destroy infected plant debris.',
    'Potato___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Raspberry___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Soybean___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Squash___Powdery_mildew': 'Improve air circulation. Apply fungicides if necessary. Use resistant varieties.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Improve air circulation. Apply fungicides if necessary.',
    'Strawberry___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.',
    'Tomato___Bacterial_spot': 'Use disease-free seeds/transplants. Apply copper-based bactericides. Rotate crops.',
    'Tomato___Early_blight': 'Use resistant varieties. Rotate crops. Apply fungicides preventatively and at first signs of disease.',
    'Tomato___Late_blight': 'Use resistant varieties. Apply fungicides preventatively and at first signs of disease. Destroy infected plant debris.',
    'Tomato___Leaf_Mold': 'Improve air circulation. Reduce humidity. Apply fungicides. Use resistant varieties.',
    'Tomato___Septoria_leaf_spot': 'Remove infected lower leaves. Rotate crops. Apply fungicides.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray with insecticidal soap or horticultural oil. Introduce predatory mites. Maintain plant vigor.',
    'Tomato___Target_Spot': 'Use resistant varieties. Rotate crops. Apply fungicides.',
    'Tomato___Tomato_mosaic_virus': 'No cure. Remove and destroy infected plants. Sanitize tools. Control insect vectors.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure. Remove infected plants. Control whitefly vectors with insecticides and exclusion.',
    'Tomato___healthy': 'Maintain good cultural practices: proper watering, fertilization, and pruning.'
}

plant_disease_map = {}
for class_label in CLASS_LABELS:
    plant_type = class_label.split('___')[0]
    if plant_type not in plant_disease_map:
        plant_disease_map[plant_type] = []
    plant_disease_map[plant_type].append(class_label)


# --- 3. Model Architecture (Must match the trained model) ---
@st.cache_resource # Cache the model loading for efficiency
def load_my_model(weights_path='best_model.weights.h5'):
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    input_tensor = Input(shape=input_shape)

    base_model = MobileNetV2(input_shape=input_shape,
                               include_top=False,
                               weights='imagenet')
    base_model.trainable = False

    x = base_model(input_tensor, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(len(CLASS_LABELS), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(weights_path)
    return model

# --- 4. Prediction Function (Adapted for Streamlit) ---
def predict_disease_streamlit(uploaded_file, model, class_labels, img_height, img_width, selected_plant_type=None):
    img = image.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    raw_predictions = model.predict(img_array)
    score = tf.nn.softmax(raw_predictions[0]).numpy()

    # Apply filtering based on selected_plant_type
    if selected_plant_type and selected_plant_type != 'All Plants' and plant_disease_map:
        allowed_diseases = plant_disease_map.get(selected_plant_type, [])
        if allowed_diseases:
            filtered_score = np.full_like(score, 1e-10) # Set non-allowed to very small number
            allowed_indices = [class_labels.index(d) for d in allowed_diseases if d in class_labels]
            for idx in allowed_indices:
                filtered_score[idx] = score[idx]
            score = filtered_score
            st.info(f"Predictions filtered for plant type: **{selected_plant_type}**")
        else:
            st.warning(f"No known diseases for selected plant type '{selected_plant_type}'. Predictions not filtered.")

    predicted_class_index = np.argmax(score)
    predicted_class_name = class_labels[predicted_class_index]
    confidence = np.max(score) * 100

    return predicted_class_name, confidence, score

# --- 5. Streamlit App Layout ---
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 AI Plant Disease Detector")
st.write("Upload an image of a plant leaf to get a disease diagnosis and solution.")

# Load the model
# Ensure best_model.weights.h5 is in the same directory as this script when deployed
with st.spinner('Loading Deep Learning Model...'):
    model = load_my_model()
st.success('Model loaded successfully!')

# Plant Type Selection
plant_options = ['All Plants'] + sorted(list(plant_disease_map.keys()))
selected_plant_type = st.selectbox(
    "First, select the plant type:",
    plant_options,
    index=0 # Default to 'All Plants'
)

# Image Uploader
uploaded_file = st.file_uploader("Upload a leaf image here:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Leaf Image', use_column_width=True)
    st.write("")

    # Make Prediction
    with st.spinner('Diagnosing...'):
        predicted_class_name, confidence, all_predictions = predict_disease_streamlit(
            uploaded_file, model, CLASS_LABELS, IMG_HEIGHT, IMG_WIDTH, selected_plant_type
        )

        solution = disease_solutions.get(predicted_class_name,
                                        "No specific solution found for this disease. Please consult an agricultural expert.")

        st.subheader("Diagnosis Result:")
        st.markdown(f"**Predicted Disease:** <span style='color:green;'>**{predicted_class_name}**</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** **{confidence:.2f}%**")
        st.info(f"**Recommended Solution:** {solution}")

        st.subheader("Top 5 Predictions:")
        top_n = 5
        top_indices = np.argsort(all_predictions)[::-1][:top_n]

        # Create lists for plotting
        top_labels_plot = [CLASS_LABELS[i] for i in top_indices]
        top_confidences_plot = [all_predictions[i] * 100 for i in top_indices]

        # Create the bar chart for top predictions
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_labels_plot[::-1], top_confidences_plot[::-1], color='lightgreen') # Reverse for highest on top
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Top 5 Predicted Diseases')
        ax.set_xlim(0, 100)
        st.pyplot(fig)

