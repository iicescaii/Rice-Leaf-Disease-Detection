import streamlit as st
import tensorflow as tf
import numpy as np
import os
from ultralytics import YOLO
import tempfile
from PIL import Image

# Load the binary rice leaf detection model
binary_model = YOLO("best.pt")

# Load the rice disease detection model
disease_model = tf.keras.models.load_model("trained_rice_disease_model_with_threshold.keras")

# Function to check if the image is a rice leaf
def is_rice_leaf(test_image, threshold=0.6):
    """
    Check if the uploaded image is a rice leaf using the binary YOLO model.
    """
    try:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(test_image.read())
            temp_image_path = temp_file.name

        # Run the YOLO model
        results = binary_model.predict(source=temp_image_path, conf=threshold, save=True)

        # Get the latest prediction folder
        predictions_dir = os.path.join(os.getcwd(), "runs/detect")
        latest_folder = max(
            [os.path.join(predictions_dir, d) for d in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, d)) and d.startswith("predict")],
            key=os.path.getmtime,
        )

        # Get the predicted image path with bounding boxes drawn
        detected_image_path = os.path.join(latest_folder, os.path.basename(temp_image_path))
        os.remove(temp_image_path)

        # Check YOLO results
        rice_leaf_detected = False
        rice_leaf_confidence = 0.0

        if results[0].boxes:  # Ensure detections exist
            for box in results[0].boxes:
                label = int(box.cls)  # Class index
                confidence = float(box.conf)  # Confidence score
                class_name = results[0].names[label].strip().lower()  # Class name (lowercase for consistency)

                # Match for "rice-leaf"
                if class_name == "rice-leaf" and confidence >= threshold:
                    rice_leaf_detected = True
                    rice_leaf_confidence = confidence
                    break

        return rice_leaf_detected, detected_image_path, rice_leaf_confidence

    except Exception as e:
        return False, None, str(e)


# Function to predict the disease
def model_prediction(test_image, threshold=0.5):
    """
    Predict the disease type for a rice leaf using the disease detection model.
    """
    try:
        # Read the uploaded image into PIL format
        image = Image.open(test_image)
        image = image.convert("RGB")  # Ensure it's in RGB format

        # Resize and preprocess the image
        image = image.resize((256, 256))
        input_arr = np.array(image) / 255.0  # Normalize image
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Get predictions
        predictions = disease_model.predict(input_arr)

        # Get the maximum confidence score and the corresponding index
        max_confidence = float(np.max(predictions))  # Ensure max_confidence is a float
        if max_confidence < threshold:
            return -1, max_confidence  # Return -1 for low-confidence predictions
        predicted_index = int(np.argmax(predictions))  # Ensure predicted_index is an integer
        return predicted_index, max_confidence
    except Exception as e:
        return -1, str(e)


    


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Rice Leaf Disease Detection"])

# Main Page
if app_mode == "Home":
    st.header("RICE LEAF DISEASE DETECTION SYSTEM")
    image_path = "homepage.png"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Rice Leaf Disease Detection System! 🌾🔍

    Our mission is to provide farmers and agricultural experts with a powerful tool to identify diseases in rice quickly and accurately. By using machine learning, our system can analyze images of rice leaves and detect potential diseases with high precision.

    ### How It Works
    1. **Upload Image**: Go to the **Rice Leaf Disease Detection** page and upload an image of a rice leaf with suspected diseases.
    2. **Analysis**: Our system will process the image and identify any diseases present.
    3. **Results**: View the results of the analysis, including the type of disease detected and the confidence level of the prediction.

    ### Why Choose Us?
    - **Accuracy:** Our system provides accurate and reliable results, helping farmers make informed decisions.
    - **User-Friendly:** Easy-to-use interface makes it simple to upload images and view results.
    - **Fast and Efficient:** Fast processing times ensure that results are available quickly.

    ### Get Started
    Click on the **Rice Leaf Disease Detection** page in the sidebar to begin using our system. Upload an image and see the results for yourself!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### About the Dataset
    This dataset is designed for **rice leaf disease detection**, an essential task in agriculture to identify and mitigate potential threats to rice crops. The dataset contains high-quality, labeled images of rice leaves, with a focus on different types of diseases. It is sourced from Kaggle and is widely used in machine learning and deep learning tasks for classification and detection.
    ---

    ### Source
    - **Kaggle Dataset**: [Rice Leaf Diseases Detection](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection)
    - The dataset is publicly available for research and educational purposes.

    ---

    ### Dataset Structure
    The dataset is organized into three main folders:
    1. **Train**: Contains **8,249 images** used for training the model. These images are labeled across various disease categories.
    2. **Validation**: Contains **1,764 images**, which are used to tune the model's hyperparameters and evaluate its performance during training.
    3. **Test**: Contains **1,777 images**, which are used for final evaluation and testing of the model to ensure generalization to unseen data.

    ---
        ### Content and Labels
    The dataset includes rice leaf images categorized into the following disease types:
    1. **Bacterial Leaf Blight**
    2. **Brown Spot**
    3. **Healthy Rice Leaves**
    4. **Leaf Blast**
    5. **Leaf Scald**
    6. **Narrow Brown Leaf Spot**
    7. **Neck Blast**
    8. **Rice Hispa**
    9. **Sheath Blight**

    Each folder (train, validation, test) contains subfolders representing these disease categories, with images specific to that label.

    ---
        ### Key Features
    - **Image Count**:
    - **Training**: 8,249 images
    - **Validation**: 1,764 images
    - **Test**: 1,777 images
    - **Image Dimensions**: The images are in various resolutions, which can be resized (e.g., to 180x180 or 224x224) for training models.
    - **Format**: All images are stored in `.jpg` format and are compatible with TensorFlow/Keras pipelines.

    ---

    ### Applications
    This dataset can be applied in:
    1. **Machine Learning**:
    - Build traditional classification models with feature extraction.
    2. **Deep Learning**:
    - Train Convolutional Neural Networks (CNNs) for multi-class classification.
    - Use transfer learning approaches (e.g., MobileNet, ResNet).
    3. **Agricultural Technology**:
    - Real-world deployment for early disease detection in rice crops.
    - Support farmers in monitoring crop health efficiently.

    ---

""")

elif app_mode == "Rice Leaf Disease Detection":
    st.header("Rice Leaf Disease Detection")

    # Fixed confidence threshold
    threshold = 0.5
            # Detailed Description for Predicted Class
    class_info = {
                "Bacterial Leaf Blight": """
                **Cause**:
                Bacterial Leaf Blight is caused by the bacteria *Xanthomonas oryzae*. It thrives in warm, humid climates.

                **Symptoms**:
                - Water-soaked lesions on leaves.
                - Yellowish streaks leading to wilting.

                **Solutions**:
                - Use resistant varieties.
                - Ensure proper drainage to reduce standing water.
                - Apply copper-based bactericides.
                - Avoid excessive nitrogen fertilization.
                """,
                
                "Brown Spot": """
                **Cause**:
                Brown Spot is caused by the fungus *Cochliobolus miyabeanus*. It thrives in areas with poor soil fertility and prolonged drought stress.

                **Symptoms**:
                - Small, dark brown spots with yellow halos.
                - Spots may merge, causing leaf blight.

                **Solutions**:
                - Use resistant varieties.
                - Apply fungicides like Mancozeb or Propiconazole.
                - Maintain balanced soil fertility.
                - Improve field drainage to reduce humidity.
                """,
                
                "Healthy Rice Leaf": """
                **Healthy Condition**:
                Represents unaffected, disease-free rice leaves.

                **Solutions for Prevention**:
                - Regular crop monitoring.
                - Use high-quality seeds.
                - Maintain proper soil health and irrigation practices.
                """,
                
                "Leaf Blast": """
                **Cause**:
                Leaf Blast is caused by the fungus *Magnaporthe oryzae*. It spreads through spores, especially in wet, humid conditions.

                **Symptoms**:
                - Small, water-soaked lesions on leaves.
                - Lesions expand into diamond-shaped spots with gray centers.
                - Severe infections can cause leaf wilting.

                **Solutions**:
                - Use resistant varieties like IR64 or Ciherang.
                - Avoid over-irrigation and maintain proper spacing.
                - Apply fungicides like Tricyclazole or Isoprothiolane.
                """,
                
                "Leaf scald": """
                **Cause**:
                Leaf Scald is caused by the bacteria *Rhynchosporium oryzae*. It appears during hot, humid conditions.

                **Symptoms**:
                - Elongated yellow lesions with dry, scalded appearances.
                - Affected leaves may curl and dry prematurely.

                **Solutions**:
                - Use disease-free seeds and resistant varieties.
                - Avoid waterlogging and improve drainage.
                - Apply bactericides to manage infection.
                """,
                
                "Narrow Brown Leaf Spot": """
                **Cause**:
                Narrow Brown Leaf Spot is caused by the fungus *Cercospora janseana*. It thrives in warm, humid environments.

                **Symptoms**:
                - Small, narrow, dark brown lesions on leaves.
                - Severe infections may lead to leaf drying and reduced photosynthesis.

                **Solutions**:
                - Ensure adequate potassium levels in the soil to improve plant resistance.
                - Apply fungicides like Propiconazole or Mancozeb.
                - Remove infected plant debris.
                """,
                
                "Neck_Blast": """
                **Cause**:
                Neck Blast is caused by the fungus *Magnaporthe oryzae*. It affects the neck or collar region of the rice plant during the reproductive stage.

                **Symptoms**:
                - Lesions on the neck or collar region.
                - Affected grains may not develop fully, leading to yield loss.

                **Solutions**:
                - Apply fungicides like Tricyclazole or Isoprothiolane at the panicle initiation stage.
                - Use resistant varieties and balanced nitrogen fertilization.
                """,
                
                "Rice Hispa": """
                **Cause**:
                Rice Hispa is caused by the insect pest *Dicladispa armigera*. The insect damages rice plants by feeding on the leaf tissue.

                **Symptoms**:
                - Parallel white streaks on leaves caused by scraping.
                - Reduced photosynthetic activity and stunted growth.

                **Solutions**:
                - Remove and destroy affected leaves early.
                - Apply insecticides like Chlorpyrifos or Lambda-cyhalothrin.
                - Maintain proper field hygiene and monitor for early signs of infestation.
                """,
                
                "Sheath Blight": """
                **Cause**:
                Sheath Blight is caused by the fungus *Rhizoctonia solani*. It thrives in high-humidity environments and dense plantings.

                **Symptoms**:
                - Irregular, water-soaked lesions on the sheath.
                - Lesions may merge, causing the sheath to collapse and the plant to lodge.

                **Solutions**:
                - Maintain proper spacing between plants to reduce humidity.
                - Use resistant varieties and balanced fertilization.
                - Apply fungicides like Validamycin or Propiconazole.
                """
    }
    # Upload Image
    test_image = st.file_uploader("Upload an Image:", type=["jpg", "png"])


    # Display the uploaded image, if available
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=False, width=100)

    # Always show the "Predict" button
    if st.button("Predict"):
        st.snow()
        st.write("Analyzing the image...")

        if test_image is None:
            # Warn the user if no image is uploaded
            st.warning("Please upload an image before clicking 'Predict'.")
        else:
            # Step 1: Detect if it's a rice leaf
            try:
                is_leaf, detected_image_path, binary_confidence = is_rice_leaf(test_image, threshold=0.6)

                if is_leaf:
                    st.success(f"Detected as Rice Leaf (Confidence: {binary_confidence * 100:.2f}%)")
                    
                    # Display the image with bounding boxes
                    st.image(detected_image_path, caption="Detected Image with Bounding Boxes", use_column_width=False, width=100)

                    # Step 2: Predict disease
                    result_index, confidence_score = model_prediction(test_image, threshold=0.6)

                    if isinstance(confidence_score, str):  # Check if confidence_score is an error message
                        st.error(f"Error during disease prediction: {confidence_score}")
                    else:
                        class_names = [
                            'Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast',
                            'Leaf Scald', 'Narrow Brown Leaf Spot', 'Neck Blast', 'Rice Hispa', 'Sheath Blight'
                        ]

                        if result_index != -1:  # High-confidence prediction
                            predicted_class = class_names[result_index]
                            st.success(f"Disease Prediction: {predicted_class}")
                            st.success(f"Confidence Score: {confidence_score * 100:.2f}%")

                            # Display Cause and Solutions for the Predicted Class
                            st.markdown("#### Cause and Description:")
                            if predicted_class in class_info:
                                st.markdown(class_info[predicted_class])
                            else:
                                st.warning("No detailed information available for this class.")
                        else:
                            st.warning("The model is not confident enough to classify this image.")
                            st.warning(f"Confidence Score: {confidence_score * 100:.2f}%")
                else:
                    st.error("The uploaded image is NOT a Rice Leaf.")

            except Exception as e:
                st.error(f"Error during processing: {e}")