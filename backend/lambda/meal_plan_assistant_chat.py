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
def chat():
    body = request.get_json(force=True)
    system_prompt = (
        f"Here is my meal plan context: "
        f"Dietary preferences and restrictions: {body.get('dietary_prefs','')}. "
        f"Day: {body.get('day','Monday')}. "
        f"Meal types: {body.get('meal_types','All meals')}. "
        f"Cuisine preferences: {body.get('cuisine_prefs','')}. "
        f"Generated meal plan: {body.get('meal_plan','')}. "
        f"Shopping list: {body.get('shopping_list','')}. "
        f"Please help me with any questions I have about this meal plan."
    )
    messages = body.get("history", []) + [{"role": "user", "content": body.get("message", "")}]
    payload = {
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 2048,
        "system": system_prompt, "messages": messages
    }
    def stream():
        for event in bedrock.invoke_model_with_response_stream(modelId=MODEL, body=json.dumps(payload))["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta" and chunk["delta"].get("type") == "text_delta":
                yield chunk["delta"]["text"]
    return Response(stream_with_context(stream()), content_type="text/plain; charset=utf-8", headers=CORS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
