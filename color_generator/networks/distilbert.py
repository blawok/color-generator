import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class Distilbert(nn.Module):
    def __init__(self, transformer_model=DistilBertModel, freeze=True):
        super(Distilbert, self).__init__()

        self.config = DistilBertConfig()
        self.config.output_hidden_states = False
        self.config.output_attentions = False
        self.transformer_model = transformer_model.from_pretrained(
            "distilbert-base-uncased", config=self.config
        )
        self.regression_head = nn.Sequential(
            nn.Linear(in_features=768, out_features=3), nn.ReLU()
        )

        if freeze:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        trans = self.transformer_model(input_ids, attention_mask)
        head = self.regression_head(trans.last_hidden_state[:, 0, :])

        return head
