import torch
from transformers import BertTokenizer
from bert_dual_output_model import BertForDualOutput  # Ensure this correctly points to your model class definition

# Adjust these paths as necessary
model_weights_path = "/Users/chamodyaavishka/Desktop/EMAIL/product/model5/model_state_dict.pt"
tokenizer_path = "/Users/chamodyaavishka/Desktop/EMAIL/product/model5"

# Initialize your model
model = BertForDualOutput()

# Load the saved model weights
model.load_state_dict(torch.load(model_weights_path))

# Set model to evaluation mode
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

def predict_email(text, tokenizer, model):
    # Tokenize the email
    inputs = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    # Predict
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        logits_phishing, logits_ai_generated = model(input_ids=input_ids, attention_mask=attention_mask)

    # Process logits
    phishing_pred = torch.softmax(logits_phishing, dim=1)
    ai_generated_pred = torch.softmax(logits_ai_generated, dim=1)

    # Convert predictions to labels
    phishing_label = torch.argmax(phishing_pred, dim=1).item()
    ai_generated_label = torch.argmax(ai_generated_pred, dim=1).item()

    return phishing_label, ai_generated_label

# Example usage
print("Enter the full email content and press Enter to predict:")
email_text = input()
phishing_label, ai_generated_label = predict_email(email_text, tokenizer, model)

print(f"Phishing Prediction: {'Phishing' if phishing_label == 1 else 'Not Phishing'}")
print(f"AI-Generated Prediction: {'NLP' if ai_generated_label == 1 else 'Not NLP'}")
