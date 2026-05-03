import json, boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "POST,OPTIONS"}

@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS)

@app.route("/", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    prompt = (
        f"You are a helpful meal planning assistant. Based on the meal plan below, generate a comprehensive grocery shopping list.\n\n"
        f"Meal plan: {body.get('meal_plan','')}\n"
        f"Day: {body.get('day','Monday')}\n"
        f"Meal Types: {body.get('meal_types','All meals')}\n"
        f"Dietary preferences and restrictions: {body.get('dietary_prefs','')}\n\n"
        f"Organize by grocery store section (Produce, Proteins, Dairy, Grains and Bread, Pantry Staples, Frozen, Other). "
        f"Consolidate duplicates, include estimated quantities. Make it easy to follow on a phone."
    )
    payload = {
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 2048,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }
    def stream():
        for event in bedrock.invoke_model_with_response_stream(modelId=MODEL, body=json.dumps(payload))["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta" and chunk["delta"].get("type") == "text_delta":
                yield chunk["delta"]["text"]
    return Response(stream_with_context(stream()), content_type="text/plain; charset=utf-8", headers=CORS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
