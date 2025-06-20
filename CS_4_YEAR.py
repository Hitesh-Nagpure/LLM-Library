# --------------------------IMPORTING MODULES---------------------------------
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# ---------------------------------------API KEY ALLOCATION----------------------------------------------
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------------------------TO EXTRACT TEXTS FROM PAGES--------------------------------------------

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# ---------------------------------TO CONVERET TEXT INTO CHUNKS OF TEXT--------------------------------------------

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# ----------------------------------TO CONVERET TEXT TO VECTORS-----------------------------------------------------

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# -------------------------------------TO CREATE A CHAIN AND ASSIGN PROMPT-----------------------------------------------

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
 
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=1)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# --------------------------------TO SEARCH FOR SIMILARITY IN USER INPUT AND STORED VECTORS---------------------------------------

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    if user_question:
        print(response)
        st.write("Answer:\n ", response["output_text"])


# ------------------------------------------------------MAIN FUNCTION------------------------------------------------------------

def main():
    # st.set_page_config("Upload Files")
    st.header("ASK QUESTION")

    user_question = st.text_input("Ask anything")

    

    # --------------------------------------------------TO CREATE SIDEBAR--------------------------------------------------------
    
    if user_question == "123":
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader( "Upload your PDF Files and Click on the Submit Button",accept_multiple_files=True, label_visibility="visible")
            if st.button("Submit"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    else:
        if user_question:
            user_input(user_question)

        


# --------------------------------------------ACTUAL EXECUTION OF PROGRAM---------------------------------------------------------

if __name__ == "__main__":
    main()