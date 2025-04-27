import boto3
import json
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

client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text")
    points = event["queryStringParameters"]["points"]
    if text and points:
        titan_config = get_titan_config(text, points)
        response = client.invoke_model(
            body = titan_config,
            modelId="amazon.titan-text-express-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        result = response_body.get("results")[0]
        return{
            "statusCode": 200,
            "body": json.dumps({"summary": result.get("outputText")}),
        }
    return {
        "statusCode": 400,
        "body": json.dumps({"error": "text and points required"}),
    }



def get_titan_config(text: str, points: str):

    prompt = f"""Text: {text} \n
        who is vincent describe in  {points} points.\n
    """

    return json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1,
            },
        }
    )