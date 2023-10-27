from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, BertTokenizer

app = Flask(__name__)

xlm_roberta_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

def whitespace_tokenizer(text: str) -> int:
    return len(text.split())

def xlm_roberta_tokenizer_func(text: str) -> int:
    return len(xlm_roberta_tokenizer(text)["input_ids"])

def bert_tokenizer_func(text: str) -> int:
    return len(bert_tokenizer(text)["input_ids"])

TOKENIZERS = {
    "whitespace": whitespace_tokenizer,
    "xlm-roberta-large": xlm_roberta_tokenizer_func,
    "bert-large-uncased": bert_tokenizer_func
}

def count_tokens(text: str, tokenizer_name: str) -> int:
    tokenizer = TOKENIZERS.get(tokenizer_name, whitespace_tokenizer)
    return tokenizer(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count', methods=['POST'])
def count():
    text = request.json.get('text', '')
    tokenizer_name = request.json.get('tokenizer_name', 'whitespace')
    return jsonify({"count": count_tokens(text, tokenizer_name)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
