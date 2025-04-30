import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockLLM
#from langchain_core.prompts import ChatMessagePromptTemplate
# import json
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


def invoke_model():
    response = model.invoke("What is the highest mountain in the world?")
    print(response)


def first_chain():
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short description for the product provided by the user"
            ),

            (
                "human",
                "{product_name}"
            )
        ]
    )
    chain = template.pipe(model)

    response = chain.invoke({
        "product_name": "bicycle"
    })
    print(response)


first_chain()
