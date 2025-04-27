from image import handler
import json

event = {
    "body": json.dumps({"description": "Abeautiful sunset"})
}

response = handler(event, {})

print(response)