from flask import Flask, request, jsonify
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
RULES_DIR = '/app/rules'

# Charger Qwen
model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


@app.route('/rules', methods=['GET'])
def get_rules():
    rules = []
    for filename in os.listdir(RULES_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(RULES_DIR, filename), 'r') as f:
                rules.append(json.load(f))
    return jsonify(rules)

@app.route('/rules', methods=['POST'])
def add_rule():
    rule = request.get_json()
    filename = f"{rule['name']}.json"
    with open(os.path.join(RULES_DIR, filename), 'w') as f:
        json.dump(rule, f)
    return jsonify({"message": "Rule added successfully"}), 201

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)