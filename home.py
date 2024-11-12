import streamlit as st

#Add an image
st.image(r"C:\Users\DAVID-PC\Pictures\Telco Churn Image.png")

def home_page():

    st.title("Embedded an ML model in GUIs --Used Streamlit")

    st.markdown("""
    This app uses machine learning to classify whether a customer is likely to be churned or not
    """)

    st.subheader("Instructions")
    st.markdown("""
    -Upload a csv file
    -Select the features for classification
    -Choose a machine learning model from the dropdown
    -Click on 'Classify' to get the predicted results
    -The app gives you a report on the performance of the model
    -Expect it to give metrics like f1, recall, precision and accuracy.
    """)

    st.header("App Features")
    st.markdown("""
    -**Data View**: Access the customer data.
    -**Predict View**: Shows the various models and predictions you will make.
    -**Dashboard**: Shows data visualisations for insights.
    """)

    st.subheader("User Benefits")
    st.markdown("""
    -**Data Driven Decisions**: You make an informed decision backed by data.
    -**Access Machine Learning**: Utilise machine learning algorithms
    """)

    st.write("#### How to Run the application")
    with st.container(border=True):
        st.code("""
        # Activate the virtual environment
        env/scripts/activate
                
        # Run the App
        streamlit run home.py
                """)
        
    # Adding the embeded link
    st.video("https://www.youtube.com/watch?v=-IM3531b1XU", autoplay=True)

    #Adding the clickable link
    st.markdown("[Watch a Demo](https://www.youtube.com/watch?v=-IM3531b1XU)")

    st.divider()
    st.write("+++" * 15)

    st.write("Need Help?")
    st.write("Contact me on:")


