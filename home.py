import streamlit as st
from PIL import Image

# Add an image
image = Image.open("assets/Telco Churn Image.png")
st.image(image, width= 500)

def home_page():
    st.title("Embedded an ML model in GUIs --Used Streamlit")

    st.markdown("""
    This app uses machine learning to classify whether a customer is likely to churn or not.
    """)

    st.subheader("Instructions")
    st.markdown("""
    - Upload a CSV file.  
    - Select the features for classification.  
    - Choose a machine learning model from the dropdown.  
    - Click on 'Classify' to get the predicted results.  
    - The app provides a report on the performance of the model.  
    - Expect metrics like F1, recall, precision, and accuracy.
    """)

    st.header("App Features")
    st.markdown("""
    - **Data View**: Access the customer data.  
    - **Predict View**: Shows the various models and predictions you will make.  
    - **Dashboard**: Shows data visualizations for insights.
    """)

    st.subheader("User Benefits")
    st.markdown("""
    - **Data-Driven Decisions**: Make informed decisions backed by data.  
    - **Access Machine Learning**: Utilize machine learning algorithms.
    """)

    st.write("#### How to Run the application")
    st.code("""
    # Activate the virtual environment
    env/scripts/activate
                
    # Run the App
    streamlit run home.py
    """)

    # Adding the embedded video link
    st.video("https://www.youtube.com/watch?v=-IM3531b1XU", autoplay=True)

    # Adding the clickable link
    st.markdown("[Watch a Demo](https://www.youtube.com/watch?v=-IM3531b1XU)")

    st.write("---" * 15)

    st.write("Need Help?")
    st.write("Contact me on:")

