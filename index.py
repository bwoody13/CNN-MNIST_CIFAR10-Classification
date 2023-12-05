import sys
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.nn.functional import softmax
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data.dataloader import DataLoader
from streamlit_drawable_canvas import st_canvas
from MNIST.mnist_cnn import MNISTCNN
from save_load import CIFAR10_type, MNIST_type, load_model, load_state_dict
from train_test import test


run_tests = '--test' in sys.argv

# Load the MNIST model
mnist_model = MNISTCNN()
load_state_dict(mnist_model, "99_52-Test", MNIST_type)
mnist_model.eval()

# Load the CIFAR10 model
cifar10_model = load_model("cifar10_res_net_v2", CIFAR10_type)
cifar10_model.eval()

# Streamlit Interface Setup
st.title("MNIST & CIFAR10 Prediction App")
st.write("Draw a digit in the canvas below for MNIST or upload an image for CIFAR10.")

col1, col2 = st.columns(2)

with col1:
    st.header("MNIST")

    # Run test set on model
    if run_tests:
        with st.spinner('Testing MNIST...'):
            print("Testing MNIST")
            data_mean = 0.1307
            data_std = 0.3081
            init_trans = transforms.Compose([
                ToTensor(),
                Normalize((data_mean,), (data_std,))
            ])
            test_ds = MNIST(root='data/', train=False, download=False,
                            transform=init_trans)
            test_loader = DataLoader(test_ds, mnist_model.batch_size)
            test(mnist_model, test_loader)

    # Create a canvas to draw on
    st.write("Draw below to access a prediction for your digit between 0 and 9. Note to get more accurate results draw images in the same format and sizing as MNIST dataset has.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Canvas fill color
        stroke_width=32,  # Increased stroke width for better scaling
        stroke_color="#FFFFFF",  # Pen color
        background_color="#000000",  # Canvas background
        height=280,  # Increased canvas height for better drawing
        width=280,  # Increased canvas width for better drawing
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
            canvas_image = canvas_image.resize((28, 28), Image.LANCZOS)

            # Resize to 28x28 for MNIST model
            st.image(canvas_image, caption='Resized Image')
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

                probabilities = softmax(prediction, dim=1)[0] * 100

                probs_df = pd.DataFrame({
                    'Digit': list(mnist_model.classes),
                    'Probability (%)': [f"{prob:.2f}" for prob in probabilities]
                })
                probs_html = probs_df.to_html(index=False)
                st.markdown(probs_html, unsafe_allow_html=True)
        else:
            st.write("Please draw a digit on the canvas first.")

with col2:
    st.header("CIFAR10")

    # Run test set
    if run_tests:
        with st.spinner('Testing CIFAR10...'):
            print("Testing CIFAR10")
            init_trans = transforms.Compose([
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            test_ds = CIFAR10(root='data/', train=False, download=True,
                              transform=init_trans)
            test_loader = DataLoader(test_ds, cifar10_model.batch_size)
            test(cifar10_model, test_loader)

    st.write("Updload an image of an airplane, car, bird, cat, deer, dog, frog,"
             " horse, ship, or truck to get a prediction for what the image is")
    uploaded_file = st.file_uploader("Upload an image for CIFAR10 prediction",
                                     type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict CIFAR10"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            reshaped_img = image.resize((32, 32), Image.LANCZOS)
            st.image(reshaped_img, caption='Resized Image')
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

                probabilities = softmax(prediction, dim=1)[0] * 100

                probs_df = pd.DataFrame({
                    'Class': list(cifar10_model.classes),
                    'Probability (%)': [f"{prob:.2f}" for prob in probabilities]
                })
                probs_html = probs_df.to_html(index=False)
                st.markdown(probs_html, unsafe_allow_html=True)
        else:
            st.write("Please upload an image first.")
