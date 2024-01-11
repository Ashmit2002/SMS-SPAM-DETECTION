import streamlit as st

from joblib import dump, load

model = load('model.joblib')
vectorizer = load('vectorizer.joblib')


# Streamlit app
def main():
    st.title("Spam or Ham Email Classification")

    # Text input for user to enter an email
    user_input = st.text_area("Enter your email here:")

    if st.button("Classify"):
        # Feature extraction
        input_data_features = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_data_features)

        # Display result
        if prediction[0] == 1:
            st.success('This is a Ham mail.')
        else:
            st.error('This is a Spam mail.')

if __name__ == '__main__':
    main()