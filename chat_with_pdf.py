import os

import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()


def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase


def main():
    st.title("Chat with your PDF 💬")
    
    query = st.text_input('Ask a question to the PDF')
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Create the knowledge base object
        knowledgeBase = process_text(text)
        
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = knowledgeBase.similarity_search(query)
  
        llm = OpenAI(model_name='gpt-4')
        chain = load_qa_chain(llm, chain_type='stuff')
        
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
            print(cost)
            
        st.write(response)
        st.write(docs)
            
            
if __name__ == "__main__":
    main()
    