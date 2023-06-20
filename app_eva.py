import os
import tempfile
import sys
from io import BytesIO
from io import StringIO
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain


def home_page():
    print("""Select any one feature from above sliderbox: \n
    1. Chat with CSV \n
    2. Summarize CSV \n
    3. Analyze CSV  """)


def chat(temperature, model_name):
    print("# Talk to CSV")
    uploaded_file = "/Users/afaanansari/Desktop/gtech/CSV-AI/fishfry-locations.csv"
    if uploaded_file:
        try:
            loader = CSVLoader(file_path=uploaded_file, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)
        llm = ChatOpenAI(temperature=temperature,
                         model_name=model_name)  # 'gpt-3.5-turbo',
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever=vectors.as_retriever(),
                                         verbose=True)

        def conversational_chat(query):
            result = qa.run(query)

            return result

        user_input = "what is the fishfry?"
        output = conversational_chat(user_input)
        print(output)


def summary(model_name, temperature, top_p, freq_penalty):
    print("# Summary of CSV")
    uploaded_file = "/Users/afaanansari/Desktop/gtech/CSV-AI/fishfry-locations.csv"
    if uploaded_file is not None:
        tmp_file_path = uploaded_file
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0)
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()
            texts = text_splitter.split_documents(data)
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
            texts = text_splitter.split_documents(data)

        print("Generate Summary")

        llm = OpenAI(model_name=model_name, temperature=temperature,
                     verbose=True, max_tokens=2000)
        chain = load_summarize_chain(llm, chain_type="stuff")
        # search = docsearch.similarity_search(" ")
        summary = chain.run(input_documents=texts[:20])
        print(summary)


def analyze(temperature, model_name):
    print("# Analyze CSV")
    uploaded_file = "/Users/afaanansari/Desktop/gtech/CSV-AI/fishfry-locations.csv"
    if uploaded_file is not None:
        tmp_file_path = uploaded_file
        df = pd.read_csv(tmp_file_path)

        def agent_chat(query):
            # Create and run the CSV agent with the user's query
            try:
                agent = create_pandas_dataframe_agent(ChatOpenAI(
                    temperature=temperature, model_name=model_name), df, verbose=True, max_iterations=4)
                result = agent.run(query)
            except:
                result = "Try asking quantitative questions about structure of csv data!"
            return result
        user_input = "how many rows in my file ?"
        output = agent_chat(user_input)

def main():

    if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
        user_api_key = os.environ["OPENAI_API_KEY"]
    else:
        user_api_key = "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"

    os.environ["OPENAI_API_KEY"] = user_api_key
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
    max_tokens = {"gpt-4": 7000, "gpt-4-32k": 31000, "gpt-3.5-turbo": 3000}
    model_name = MODEL_OPTIONS[0]
    top_p = 0
    freq_penalty = 0.0
    temperature = 0.0
    functions = [
        "home",
        "Chat with CSV",
        "Summarize CSV",
        "Analyze CSV",
    ]
    selected_function = functions[1]
    if selected_function == "home":
        home_page()
    elif selected_function == "Chat with CSV":
        chat(temperature=temperature, model_name=model_name)
    elif selected_function == "Summarize CSV":
        summary(model_name=model_name, temperature=temperature,
                top_p=top_p, freq_penalty=freq_penalty)
    elif selected_function == "Analyze CSV":
        analyze(temperature=temperature, model_name=model_name)


if __name__ == "__main__":
    main()
