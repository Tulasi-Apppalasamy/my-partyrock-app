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
        f"You are a professional nutritionist and meal planning expert. Generate a detailed, practical meal plan.\n\n"
        f"Day: {body.get('day','Monday')}\nDiet Type: {body.get('diet_type','Vegetarian')}\n"
        f"Meal types to include: {body.get('meal_types','All meals')}\n"
        f"Dietary preferences and restrictions: {body.get('dietary_prefs','')}\n"
        f"Cuisine preferences and favorite ingredients: {body.get('cuisine_prefs','')}\n\n"
        f"For each meal include: dish name, brief description, prep/cook time, key ingredients. Format clearly with meals as headers."
    )
    content = [{"type": "text", "text": prompt}]
    if body.get("file_data"):
        t = "image" if body.get("file_mime", "").startswith("image/") else "document"
        content.insert(0, {"type": t, "source": {"type": "base64", "media_type": body["file_mime"], "data": body["file_data"]}})
    payload = {
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [{"role": "user", "content": content}]
    }
    def stream():
        for event in bedrock.invoke_model_with_response_stream(modelId=MODEL, body=json.dumps(payload))["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta" and chunk["delta"].get("type") == "text_delta":
                yield chunk["delta"]["text"]
    return Response(stream_with_context(stream()), content_type="text/plain; charset=utf-8", headers=CORS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
