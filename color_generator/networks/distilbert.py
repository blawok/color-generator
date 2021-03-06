"""Define DistilBERT network function."""
from typing import Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow import squeeze, int32
from transformers import TFDistilBertModel, DistilBertConfig


def distilbert() -> Model:

    qa = Input((32,), dtype=int32)
    qa_mask = Input((32,), dtype=int32)

    config = DistilBertConfig()
    config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased',
                                                          config=config)

    qa_embedding = transformer_model(qa, attention_mask=qa_mask)[0]
    cls_encodings = squeeze(qa_embedding[:, 0:1, :], axis=1)
    output_dense = Dense(3, activation='sigmoid')(cls_encodings)

    model = Model(inputs=[qa, qa_mask], outputs=output_dense)
    return model

