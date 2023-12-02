import streamlit as st
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
from utils import scale_and_center_digit

# Load the MNIST model from the .pth file
mnist_model = torch.load("MNIST/models/99_43-Test.pth", map_location=torch.device('cpu'))
mnist_model.eval()  # Set the model to evaluation mode
cifar10_model = torch.load("CIFAR10/models/first_trained_model.pth", map_location=torch.device("cpu"))
cifar10_model.eval()

# Streamlit Interface Setup
st.title("MNIST & CIFAR10 Prediction App")
st.write("Draw a digit in the canvas below for MNIST or upload an image for CIFAR10.")

col1, col2 = st.columns(2)

with col1:
    st.header("MNIST")
    # Create a canvas to draw on
    st.write("draw below to access a prediction for your digit between 0 and 9")
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

            # Center image
            canvas_image = scale_and_center_digit(canvas_image)

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

with col2:
    st.header("CIFAR10")
    st.write("Updload an image of an airplane, car, bird, cat, deer, dog, frog, horse, ship, or truck "
             "to get a prediction for what the image is")
    uploaded_file = st.file_uploader("Upload an image for CIFAR10 prediction", type=['png', 'jpg', 'jpeg'])

    if st.button("Predict CIFAR10"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            # Resize and preprocess the image
            preprocess = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            input_tensor = preprocess(image).unsqueeze(0)

            # Make a prediction
            with torch.no_grad():
                cifar10_model.eval()
                prediction = cifar10_model(input_tensor)
                predicted_class = torch.argmax(prediction).item()
                st.write(f"Predicted CIFAR10 Class: {cifar10_model.classes[predicted_class]}")
        else:
            st.write("Please upload an image first.")