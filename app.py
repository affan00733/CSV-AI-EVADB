import evadb
from evadb.interfaces.relational.db import connect
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv.main import load_dotenv
load_dotenv()

if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
    user_api_key = os.environ["OPENAI_API_KEY"]
else:
    user_api_key = "API_KEY"
os.environ["OPENAI_API_KEY"] = user_api_key

conn = connect()
cursor = conn.cursor()
cursor.drop_table("MyCSV").execute()
create_table_query = f"""
    CREATE TABLE IF NOT EXISTS MyCSV(
        venue_name TEXT(30),
        venue_type TEXT(30),
        venue_address TEXT(30),
        website TEXT(30),
        menu_url TEXT(30),
        menu_text TEXT(30),
        phone TEXT(30),
        email TEXT(30),
        alcohol TEXT(30),
        lunch TEXT(30)
    );
    """
cursor.query(create_table_query).df()
cursor.load(file_regex="./fishfry-locations.csv",
            format="csv", table_name="MyCSV").execute()

df = cursor.table("MyCSV").df()
df['combined'] = df[df.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

cursor.drop_udf("embedding").execute()
embedding_udf = cursor.create_udf(
    udf_name="embedding",
    if_not_exists=True,
    impl_path=f"./sentence_feature_extractor.py",
)
embedding_udf.execute()

loader = DataFrameLoader(df, page_content_column="combined")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
vectors = FAISS.from_documents(docs, embeddings)

llm = ChatOpenAI(temperature=0.9,
                 model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=vectors.as_retriever(),
                                 verbose=True)
user_input = str(input("Enter Quesiton?: \n"))
print(qa.run(user_input))
