
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS,Chroma
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import tempfile
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
import textwrap
import tiktoken

API_KEY = st.secrets['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"] = API_KEY
    # # print('Hello')
st.set_page_config(page_title="PDF Summarizer")
st.header("Summarize your PDF files")
# load_dotenv()
uploaded_file = st.file_uploader('Upload your files', type=(['pdf']))

temp_file_path = os.getcwd()
while uploaded_file is None:
    x = 1      

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load_and_split()
    st.write(len(documents))

    llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k")#gpt-3.5-turbo-16k
    # text_splitter = CharacterTextSplitter(
    # separator = "\n",
    # chunk_size = 1000,
    # chunk_overlap  =    0, #striding over the text
    # length_function = len)
    # docs = text_splitter.split_documents(documents)
    
    # Tokens details
    tokenizer = tiktoken.get_encoding('cl100k_base')
    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    token_counts = [tiktoken_len(doc.page_content) for doc in documents]
    st.write(token_counts)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=llm.get_num_tokens,
    separators=['\n\n', '\n', ' ', ''] 
    )

    chunks = text_splitter.split_documents(documents)
    st.write(len(chunks))
    total_len=0
    # st.write(chunks[0].page_content)
    # st.write(tiktoken_len(chunks[0]), tiktoken_len(chunks[1]),tiktoken_len(chunks[2]))
    for i in range(len(chunks)):
        st.write(i)
        st.write(tiktoken_len(chunks[i].page_content))
        total_len+=tiktoken_len(chunks[i].page_content)
    st.write(total_len)    

    # chunks = text_splitter.split_text(documents[1].page_content)
    # st.write(len(chunks))
    # # st.write(chunks[0])
    # st.write(tiktoken_len(chunks[2]))

    # Custom prompt
    prompt_template = """Write a concise bullet point summary of the following:
    {text}
    CONSCISE SUMMARY IN BULLET POINTS:"""
    refine_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    
    chain1 = load_summarize_chain(llm, chain_type="stuff")
    chain2 = load_summarize_chain(llm, chain_type="map_reduce")
    chain3 = load_summarize_chain(llm, chain_type="refine")
    chain4 = load_summarize_chain(llm, chain_type="stuff",prompt=refine_prompt)

    col1, col2,col3,col4 = st.columns(4,gap="small")
    with st.form("Summary_Docs"):
        with col1:
            calc_1 = st.checkbox("Stuff")
        with col2:
            calc_2 = st.checkbox("Map_Reduce")
        with col3:
            calc_3 = st.checkbox("Refine")
        with col4:
            calc_4 = st.checkbox("Custom")
        submit_button = st.form_submit_button("Get Summary",use_container_width=True)
    if submit_button:
        if calc_1:
            st.text_area('Description','''Stuffing is the simplest method, where you simply stuff all the related data into the prompt as context to pass to the language model''',label_visibility='hidden')
            with st.spinner('Please wait...'):
                chain1 = load_summarize_chain(llm, chain_type="stuff")
                result=chain1.run(chunks)
                wrapped_text = textwrap.fill(result, width=100,break_long_words=False,replace_whitespace=False)
                st.write(wrapped_text)
        if calc_2:
            st.text_area('Description','''Map Reduce involves running an initial prompt on each chunk of data (for summarization tasks, this could be a summary of that chunk; for question-answering tasks, it could be an answer based solely on that chunk). Then a different prompt is run to combine all the initial outputs''',label_visibility='hidden')
            with st.spinner('Please wait...'):
                chain2 = load_summarize_chain(llm, chain_type="map_reduce")
                result=chain2.run(chunks)
                wrapped_text = textwrap.fill(result, width=100,break_long_words=False,replace_whitespace=False)
                st.write(wrapped_text)
        if calc_3:
            st.text_area('Description','''Refine involves running an initial prompt on the first chunk of data, generating some output. For the remaining documents, that output is p)assed in, along with the next document, asking the LLM to refine the output based on the new document''',label_visibility='hidden')
            with st.spinner('Please wait...'):
                chain3 = load_summarize_chain(llm, chain_type="refine")
                result=chain3.run(chunks)
                wrapped_text = textwrap.fill(result, width=100,break_long_words=False,replace_whitespace=False)
                st.write(wrapped_text)    
        if calc_4:
            st.text_area('Description','''Stuff chain with custom prompt''',label_visibility='hidden')
            with st.spinner('Please wait...'):
                chain4 = load_summarize_chain(llm, chain_type="stuff",prompt=refine_prompt)
                result=chain4.run(chunks)
                wrapped_text = textwrap.fill(result, width=100,break_long_words=False,replace_whitespace=False)
                st.write(wrapped_text)         
    
