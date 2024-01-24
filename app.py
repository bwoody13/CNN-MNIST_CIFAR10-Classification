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


is_dev = 'dev' in sys.argv

# Load the MNIST model
if "mnist_model" not in st.session_state:
    mnist_model = MNISTCNN()
    load_state_dict(mnist_model, "99_52-Test", MNIST_type)
    mnist_model.eval()
    st.session_state["mnist_model"] = mnist_model
else:
    mnist_model = st.session_state["mnist_model"]

# Load the CIFAR10 model
if "cifar10_model" not in st.session_state:
    cifar10_model = load_model("cifar10_res_net_v2", CIFAR10_type)
    cifar10_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar10_model.eval()
else:
    cifar10_model = st.session_state["cifar10_model"]

# Streamlit Interface Setup
st.title("MNIST & CIFAR10 Prediction App")
st.write("Check the test set performace by clicking on the buttons for MNIST or CIFAR10. Note this may take a while. For faster testing and running models try using the notebook in the repo.")
st.write("Draw a digit in the canvas below for MNIST or upload an image for CIFAR10.")

col1, col2 = st.columns(2)

with col1:
    st.header("MNIST")

    st.write("Click below to run the MNIST CNN on the test data loaded directly from PyTorch.")
    # Run test set on model
    if st.button("Run Model on Test Data"):
        with st.spinner('Testing MNIST...'):
            print("Testing MNIST")
            if 'm_pred_html' not in st.session_state:
                data_mean = 0.1307
                data_std = 0.3081
                init_trans = transforms.Compose([
                    ToTensor(),
                    Normalize((data_mean,), (data_std,))
                ])
                # if 'm_test_ds' not in st.session_state:
                m_test_ds = MNIST(root='data/', train=False, download=False,
                                  transform=init_trans)
                #     st.session_state['m_test_ds'] = m_test_ds
                # else:
                #     m_test_ds = st.session_state['m_test_ds']
                m_test_loader = DataLoader(m_test_ds, mnist_model.batch_size)
                loss, acc, class_acc = test(mnist_model, m_test_loader, is_dev)
                st.session_state['m_loss_acc'] = f"Test Loss: {loss:.3f}, Test Accuracy: {acc:.3f}"
                preds = pd.DataFrame(class_acc, columns=['Class', 'Accuracy'])
                m_pred_html = preds.to_html()
                st.session_state['m_pred_html'] = m_pred_html
            st.write(st.session_state['m_loss_acc'])
            st.markdown(st.session_state['m_pred_html'], unsafe_allow_html=True)
            st.markdown("---")

    # Create a canvas to draw on
    st.write("Draw below to access a prediction for your digit between 0 and 9. Note to get more accurate results draw images in the same format and sizing as MNIST dataset has.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=32,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
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

    st.write("Click below to run the CIFAR10 CNN on the test data loaded directly from PyTorch.")
    # Run test set
    if st.button("Run Model on Test Data", key="run-model-c10"):
        with st.spinner('Testing CIFAR10...'):
            print("Testing CIFAR10")
            if 'c_pred_html' not in st.session_state:
                init_trans = transforms.Compose([
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                if 'c_test_ds' not in st.session_state:
                    c_test_ds = CIFAR10(root='data/', train=False, download=False,
                                        transform=init_trans)
                    st.session_state['c_test_ds'] = c_test_ds
                else:
                    c_test_ds = st.session_state['c_test_ds']
                c_test_loader = DataLoader(c_test_ds, cifar10_model.batch_size)
                loss, acc, class_acc = test(cifar10_model, c_test_loader, is_dev)
                st.session_state['c_loss_acc'] = f"Test Loss: {loss:.3f}, Test Accuracy: {acc:.3f}"
                preds = pd.DataFrame(class_acc, columns=['Class', 'Accuracy'])
                c_pred_html = preds.to_html()
                st.session_state['c_pred_html'] = c_pred_html
            st.write(st.session_state['c_loss_acc'])
            st.markdown(st.session_state['c_pred_html'], unsafe_allow_html=True)
            st.markdown("---")

    st.write("Upload an image of an airplane, car, bird, cat, deer, dog, frog,"
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
