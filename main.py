from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFAutoModelForSeq2SeqLM, pipeline

REPO_ID = "saintrivers/summarization-tutorial"

# Create a Flask application
app = Flask('summary-bot')

# Define an endpoint for predictions
@app.route('/summarize', methods=['POST'])
def predict():
    fill_mask = request.get_json()
    text = fill_mask['text']

    # tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    # model = TFAutoModelForSeq2SeqLM.from_pretrained(REPO_ID)
    summarizer_xlsum = pipeline("summarization", model=REPO_ID)
    out = summarizer_xlsum(text)

    # Prepare the result in JSON format
    result = {
        'results': out
    }

    # Return the result as a JSON response
    return jsonify(result)

# Run the Flask application
if __name__ == "__main__":
    # Run the app in debug mode on all available network interfaces
    app.run(debug=True, host='0.0.0.0', port=9696)
