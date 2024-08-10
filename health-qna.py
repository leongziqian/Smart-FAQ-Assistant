import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key = st.secrets["mykey"]

# Replace with your embedding model
model = "text-embedding-ada-002"

# Load your dataset
try:
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    # Convert the 'Question_Embedding' column from string to actual NumPy arrays
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.array(eval(x)))
except Exception as e:
    st.error(f"Error loading CSV file: {e}")

# Function to get embedding
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding).reshape(1, -1)

def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x.reshape(1, -1), user_question_embedding))

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.6  # You can adjust this value

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer, max_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"

def main():
    st.title("Heart, Lung, and Blood Health Q&A")

    user_question = st.text_input("Ask a question about health:")
    if st.button("Get Answer"):
        if user_question:
            best_answer, similarity_score = find_best_answer(user_question)
            st.write(f"Similarity Score: {similarity_score:.2f}")
            st.write(best_answer)
            
            # Rating system
            rating = st.slider("Rate the helpfulness of the answer", 1, 5, 3)
            st.write(f"Thank you for rating this answer: {rating} star(s)")
        else:
            st.write("Please enter a question.")
            
#Additional Features (Optional)
st.sidebar.title("FAQs")
st.sidebar.write("what is are cardiomyopathy?")
st.sidebar.write("who is at risk for cardiomyopathy")

if __name__ == "__main__":
    main()
