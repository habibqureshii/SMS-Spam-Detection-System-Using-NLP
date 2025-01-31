import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))  # Trained classification model
cv = pickle.load(open('spam_vectorizer.pkl', 'rb'))  # Corrected CountVectorizer file

def main():
    # App Title and Description
    st.title("📧 Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as **Spam** or **Not Spam**.")

    # Input Section
    st.subheader("Enter the email text below:")
    user_input = st.text_area("Type or paste the email content here:", height=150)  # Input field for email content

    # Create two buttons side by side
    col1, col2 = st.columns(2)

    # Classify Button
    with col1:
     if st.button("Classify"):
        if user_input.strip():  # Check if input is not empty
            try:
                # Preprocess and predict
                data = [user_input]
                vec = cv.transform(data).toarray()  # Transform the input using the vectorizer
                result = model.predict(vec)  # Predict using the trained model

                # Display the result
                if result[0] == 0:
                    st.success("✅ This is **Not Spam**.")
                else:
                    st.error("🚨 This is **Spam**.")
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
        else:
            st.warning("⚠️ Please enter an email to classify.")
    with col2:
            # Clean Button
     if st.button("Clean"): # Clear the input field
        st.session_state["user_input"] = ""  # Reset the session state value
# Run the application
if __name__ == "__main__":
    main()
