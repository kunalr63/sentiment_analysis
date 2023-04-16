from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load the pre-trained model and tokenizer from Hugging Face model hub
model_name = "kunalr63/my_awesome_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set the device for the model (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a route for the sentiment analysis API
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Receive text input from the request
    text = request.data['text']

    # Tokenize the text and convert it to a tensor
    inputs = tokenizer(text, return_tensors='pt')
    inputs.to(device)

    model = AutoModelForSequenceClassification.from_pretrained("kunalr63/my_awesome_model")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    sentiment_label=model.config.id2label[predicted_class_id]

    # Return the predicted sentiment label as output
    return jsonify({'text':text,'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
