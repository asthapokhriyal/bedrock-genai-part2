import boto3
import json
import base64
from time import time
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

S3_BUCKET = "image-bucket-100001"

client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")
s3_client = boto3.client('s3')

def handler(event, context):
    body = json.loads(event["body"])
    description = body.get("description")
    if description:
        titan_config = get_titan_config(description)
        response = client.invoke_model(
            body=titan_config,
            modelId="amazon.titan-image-generator-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        signed_url = save_image_to_s3(base64_image)
        return{
            "statusCode": 200,
            "body": json.dumps({"url": signed_url}),
        }
    
    
def save_image_to_s3(base64_image: str):
    image_file = base64.b64decode(base64_image)
    timestamp = int(time())
    image_name = str(timestamp) + '.jpg'

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=image_name,
        Body=image_file,
    )
    signed_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET, 'Key': image_name},
        ExpiresIn=3600
    )
    return signed_url


def get_titan_config(description: str):
    return json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams":{
                "text": description,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": 8.0,
            },
        }
    )