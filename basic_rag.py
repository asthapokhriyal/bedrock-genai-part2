from langchain_aws.llms import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3
from dotenv import load_dotenv
import os

load_dotenv()
# Get credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION")

# Set up the session
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")
model = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock)


my_data = [
    "The weather is nice today.",
    "Last night's game ended in tie.",
    "Don likes to eat pizza.",
    "Don likes to eat pasta."
]

question = "What does Don likes to eat?"

AWS_REGION = "us-east-1"

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

#create vector store
vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

#create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.get_relevant_documents(question)

results_string = []
for result in results:
    results_string.append(result.page_content)

#build template:
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        (
            "user", "{input}"
        ),
    ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response)