from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from operator import itemgetter
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

# Load PDF file
loader = PyMuPDFLoader('/workspaces/uplimit_langchain/pil.3474.pdf')
data = loader.load()

# Split leaflet into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100
)
split_drug_docs = text_splitter.split_documents(data)
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.from_documents(documents=split_drug_docs, embedding=embedding_model)

# Create Prompt
qna_prompt = """
You are an AI assistant for reading information from medicine leaflets and you explain things in simple manner for a human to understand 

If you don't know the answer, simply state that you don't have enough information to provide an answer. Do not attempt to make up an answer.

If the user greets you with a greeting like "Hi", "Hello", or "How are you", respond in a friendly manner.

Use the following pieces of context to answer the user's question
question: {question}
context : {context}
Answer:
"""

# Define messages for the chatbot prompt
messages = [
    SystemMessagePromptTemplate.from_template("You're a helpful AI assistant"),
    HumanMessagePromptTemplate.from_template(qna_prompt),
]

qna_prompt_template = ChatPromptTemplate.from_messages(messages)

# Create LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# Create Retriever that adds the context
retriever = vector_store.as_retriever()

# Do the Post-processing on the output of the retriever
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG Chain
rag_chain = (
    {"context": itemgetter("input") | retriever | format_docs,
     "question": RunnablePassthrough()}
    | qna_prompt_template
    | llm
    | StrOutputParser()
)

def provide_bot_response(user_question):
    rag_response = rag_chain.invoke({"input": user_question})
    return rag_response

def main():
    rag_response = rag_chain.invoke({"input": "Can I give this medicine to my child?"})
    print (rag_response)

if __name__ == "__main__":
    main()
