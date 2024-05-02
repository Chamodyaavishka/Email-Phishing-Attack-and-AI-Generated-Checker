from transformers import BertModel
import torch.nn as nn

class BertForDualOutput(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', num_labels_1=2, num_labels_2=2):
        super(BertForDualOutput, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)

        # Classification head for the first task (e.g., phishing or not)
        self.classifier_1 = nn.Linear(self.bert.config.hidden_size, num_labels_1)
        
        # Classification head for the second task (e.g., AI-generated or not)
        self.classifier_2 = nn.Linear(self.bert.config.hidden_size, num_labels_2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get logits from each classifier
        logits_1 = self.classifier_1(pooled_output)
        logits_2 = self.classifier_2(pooled_output)

        return logits_1, logits_2
