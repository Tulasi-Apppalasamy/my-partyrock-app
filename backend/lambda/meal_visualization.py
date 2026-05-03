import json, boto3, random

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "POST,OPTIONS"}

def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}
    body = json.loads(event.get("body") or "{}")
    meal_plan = body.get("meal_plan", "a healthy kids meal")
    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": f"An appetizing, colorful image of {meal_plan}, food photography, bright, suitable for children",
            "negativeText": "watermarks, text, logos"
        },
        "imageGenerationConfig": {
            "numberOfImages": 1, "width": 1280, "height": 720,
            "seed": random.randint(0, 2147483647)
        }
    }
    resp = bedrock.invoke_model(modelId="amazon.nova-canvas-v1:0", body=json.dumps(payload))
    result = json.loads(resp["body"].read())
    return {
        "statusCode": 200,
        "headers": {**CORS, "Content-Type": "application/json"},
        "body": json.dumps({"image": result["images"][0]})
    }
