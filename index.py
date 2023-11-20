import streamlit as st
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas

# Load the MNIST model from the .pth file
mnist_model = torch.load("MNIST/models/99_29-Test.pth", map_location=torch.device('cpu'))
mnist_model.eval()  # Set the model to evaluation mode

# Streamlit Interface Setup
st.title("MNIST Digit Recognition")
st.write("Draw a digit in the canvas below and get its prediction.")

# Create a canvas to draw on
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Canvas fill color
    stroke_width=10,  # Pen width
    stroke_color="#FFFFFF",  # Pen color
    background_color="#000000",  # Canvas background
    height=150,  # Canvas height
    width=150,  # Canvas width
    drawing_mode="freedraw",
    key="canvas",
)

# Predict using the MNIST model
if st.button("Predict MNIST"):
    if canvas_result.image_data is not None:
        # Convert canvas image to PIL Image for processing
        canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        canvas_image = canvas_image.convert('L')  # Convert to grayscale

        # Resize to 28x28 for MNIST model
        canvas_image = ImageOps.fit(canvas_image, (28, 28), Image.Resampling.LANCZOS)

        # Preprocess the canvas image for the MNIST model
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = preprocess(canvas_image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            prediction = mnist_model(input_tensor)
            predicted_class = torch.argmax(prediction).item()
            st.write(f"Predicted MNIST Digit: {predicted_class}")
    else:
        st.write("Please draw a digit on the canvas first.")