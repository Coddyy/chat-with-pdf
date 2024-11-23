from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
import streamlit as st

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

# Initialize the vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",  # Where to save data locally
)

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant (Co-Pilot). Answer the question based only the data provided."),
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
    ]
)

# Define the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


######### Terminal Verison ###### [Run !python3.12 chat.py]
# Main loop
# def main():
#     while True:
#         query = input("User (or type 'q', 'quit', or 'exit' to end): ")
#         if query.lower() in ['q', 'quit', 'exit']:
#             break
        
#         result = retrieval_chain.invoke({"input": query})
#         print("Assistant: ", result["answer"], "\n\n")

# # Run the main loop
# if __name__ == "__main__":
#     main()

import time
######### UI Verison ###### [Run !python3.12 -m streamlit run chat.py]
st.set_page_config(page_title="Document Co-Pilot")
st.header("Document Co-Pilot")
question =  st.text_input("Input", key="input")
submit = st.button("Ask a question")
if submit:
    with st.spinner(":sunglasses: I'm on it, hold on a moment..."):
        response = retrieval_chain.invoke({"input": question})
        res=response['answer']
    st.header(res)


