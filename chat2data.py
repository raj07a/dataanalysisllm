import streamlit as st
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the dimension of your embeddings
dimension = 512  # This should match the dimension of your embeddings

# Initialize a FAISS index
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Add sample embeddings to the index (in practice, use real data)
sample_embeddings = np.random.random((10, dimension)).astype('float32')
index.add(sample_embeddings)

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def retrieve_information(query):
    # Simulate information retrieval
    related_info = "AI is transforming industries by automating processes and enhancing decision-making."
    return related_info

def generate_with_rag(query):
    # Step 1: Retrieve related information
    retrieved_info = retrieve_information(query)
    
    # Step 2: Combine the query with retrieved information
    combined_input = f"{retrieved_info}\n\nQuestion: {query}"
    
    # Step 3: Generate an answer using GPT-2
    input_ids = tokenizer.encode(combined_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=150)
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_answer

# Streamlit application
st.title("LLM-Based Data Analysis Tool")

query = st.text_input("Enter your query:", "How does AI impact healthcare?")

if st.button("Generate Answer"):
    answer = generate_with_rag(query)
    st.write("Generated Answer:")
    st.write(answer)

# Streamlit will automatically detect the changes and refresh the app
