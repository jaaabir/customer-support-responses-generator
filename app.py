import os
import time 
import torch
import numpy as np
import streamlit as st
from datasets import load_dataset
from hyperparameters import pretext, DATASET, max_source_length
from transformers import T5Tokenizer, T5ForConditionalGeneration

CUDA  = torch.cuda.is_available()

@st.cache_resource
def load_model_tkn():
    MODEL_PATH = os.path.join('models','t5-customer-support')
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    if not CUDA: model.to('cpu')
    return model, tokenizer

@st.cache_data
def load_questions():
    sample_questions = np.random.choice(load_dataset(DATASET)['train']['query'], 5)
    return sample_questions


def generate_response(query):
    model, tokenizer = load_model_tkn()
    query = pretext + query.lower()
    input_ids = tokenizer.encode(query, return_tensors='pt', max_length = max_source_length, truncation=True)
    if CUDA:
        input_ids = input_ids.to('cuda')
        model.to('cuda')
    output_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    st.set_page_config(page_title="Customer Support Chatbot", page_icon=":robot:", layout="wide")
    st.sidebar.title("Sample Questions")
    sample_questions = load_questions()
    selected_question = st.sidebar.selectbox("Choose a question", sample_questions)
    if selected_question:
        st.sidebar.write(f"CUDA Available: :{'green' if CUDA else 'red'}[{CUDA}]")

    st.title("Customer Support Chatbot")
    st.markdown("""
    Welcome to the Customer Support Chatbot. Ask any question related to our services, and we'll provide an accurate response.
    """)


    user_query = st.text_area("Enter your question here", selected_question if selected_question else "")

    if st.button("Submit") or st.session_state.get('submit_button', False):
        if user_query:
            st.session_state['submit_button'] = True
            with st.spinner("Generating response..."):
                response = generate_response(user_query)
                print(user_query)
                print(response)

            st.write("## Response")
            words = response.split()
            placeholder = st.empty()
            response_text = ""
            for word in words:
                response_text += word + " "
                placeholder.markdown(response_text)
                time.sleep(0.1)  
            st.session_state['submit_button'] = False
        else:
            st.warning("Please enter a question to get a response.")

if __name__ == "__main__":
    main()
