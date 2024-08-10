import streamlit as st
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity

# Load Data & Embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    return df

df = load_data()

def get_embedding(text):
  # Use OpenAI or another embedding model to generate an embedding
  response = openai.Embedding.create(input=[text])
  embedding = response['data'][0]['embedding']
  return embedding
    
# Build the Streamlit Interface
st.title("Heart, Lung, and Blood Health Q&A")

user_question = st.text_area("Ask a question about heart, lung, or blood health:", height=150)
similarity_threshold = 0.8  # Set a threshold for similarity

if st.button("Get Answer"):
    if user_question:
        # Generate an embedding for the user's question
        user_embedding = get_embedding(user_question)  # Ensure `get_embedding` is defined or remove if not needed
        
        if user_embedding is not None:
            # Calculate cosine similarities
            df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_embedding))
            
            # Find the most similar question
            best_match = df.loc[df['Similarity'].idxmax()]
            
            # Display the answer if similarity is above the threshold
            if best_match['Similarity'] >= similarity_threshold:
                st.subheader("Answer:")
                st.write(best_match['Answer'])
                st.write(f"**Similarity Score:** {best_match['Similarity']:.2f}")
            else:
                st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
        else:
            st.write("Unable to process the question. Please try again.")
    else:
        st.write("Please enter a question.")

# Additional Features (Optional)
if st.button("Clear"):
    st.text_area("Ask a question about heart, lung, or blood health:", value="", key="reset")

st.sidebar.title("FAQs")
st.sidebar.write("Common questions related to heart, lung, and blood health will be displayed here.")
