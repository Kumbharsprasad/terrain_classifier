import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from streamlit_extras.add_vertical_space import add_vertical_space


# Load the trained model
model = tf.keras.models.load_model("./terrain_one.keras")

# Define image size (ensure this matches your model's expected input size)
IMAGE_SIZE = (256, 256)  # Adjust if needed to match your model

st.title("Military Monk Terrain Classifier 🎖️🛦🪖")

# Class names for terrain classification
class_names = ['Desert', 'Forest', 'Mountain', 'Plains']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

with st.sidebar:
    st.title("Military Monk Terrain Classifier")
    add_vertical_space(2)
    st.write("Reach out to me on [LinkedIn](https://www.linkedin.com/in/prasad-kumbhar-/)")
    add_vertical_space(2)
    st.write("Our Blog [Website](https://military-monk.blogspot.com/)")


# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image in the main section
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = preprocess_image(image)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100  # Confidence percentage

    # Display results
    st.write(f"**Predicted Class**: {class_names[predicted_class]}")
    st.write(f"**Confidence**: {confidence:.2f}%")
