import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class ConceptVectorFormation(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden_size, lstm_num_layers, lstm_dropout):
        super(ConceptVectorFormation, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert_model.config.hidden_size, lstm_hidden_size, 
                            num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout)
        
    def forward(self, input_text):
        # BERT encoding
        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        outputs = self.bert_model(input_ids)
        bert_hidden_states = outputs.last_hidden_state
        
        # LSTM modeling
        lstm_output, _ = self.lstm(bert_hidden_states)
        concept_vector = lstm_output[:, -1, :]  # Take the last output of LSTM as concept vector
        
        return concept_vector

# Example usage:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
concept_model = ConceptVectorFormation('bert-base-uncased', lstm_hidden_size=256, lstm_num_layers=2, lstm_dropout=0.2)
input_text = "This is an example input text."
concept_vector = concept_model(input_text)
print(concept_vector)
