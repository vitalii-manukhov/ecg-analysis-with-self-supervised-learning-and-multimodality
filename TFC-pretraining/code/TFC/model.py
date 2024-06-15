"""
...
"""
from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer

BERT_PRETRAIN_PATH = "../../BERT_pretrain/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TFC(nn.Module):
    """
    ...
    """
    def __init__(self, configs):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned,
                                                   dim_feedforward=2*configs.TSlength_aligned,
                                                   nhead=2, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned,
                                                   dim_feedforward=2*configs.TSlength_aligned,
                                                   nhead=2,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t, x_in_f):
        """
        ...
        """
        # Use Transformer
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        # Cross-space projector
        z_time = self.projector_t(h_time)

        # Frequency-based contrastive encoder
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        # Cross-space projector
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


# Downstream classifier only used in finetuning
# class target_classifier(nn.Module):
#     def __init__(self, configs):
#         super(target_classifier, self).__init__()
#         self.logits = nn.Linear(2 * 128, 64)
#         self.logits_simple = nn.Linear(64, configs.num_classes_target)

#     def forward(self, emb):
#         emb_flat = emb.reshape(emb.shape[0], -1)
#         emb = torch.sigmoid(self.logits(emb_flat))
#         pred = self.logits_simple(emb)
#         return pred

class FrozenLanguageModel(nn.Module):
    """
    Description:
        A frozen version of the language model.
    """
    def __init__(self):
        super(FrozenLanguageModel, self).__init__()
        self.language_model = BertModel.from_pretrained(
            'emilyalsentzer/Bio_ClinicalBERT',
            cache_dir=BERT_PRETRAIN_PATH
        )
        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Description:
            Forward pass of the frozen language model.
        Args:
            input_ids: The input ids of the language model.
            attention_mask: The attention mask of the language model.
        Returns:
            The last hidden state of the language model.
        """
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_representation = outputs.last_hidden_state[:, 0, :]
        return sentence_representation


class TargetClassifier(nn.Module):
    """
    ...
    """
    def __init__(self, configs):
        super(TargetClassifier, self).__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)
        self.text_encoder = FrozenLanguageModel()
        self.embedding_dim = self.text_encoder.language_model.config.hidden_size
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir=BERT_PRETRAIN_PATH)

    def zero_shot_process_text(self, categories):
        """
        Description:
            Process the text data for zero-shot learning.

        Args:
            text_data: The text data to be processed.

        Returns:
            torch.Tensor: The processed text data.
        """
        zero_shot_text_prompt = "The ECG of {label}, a type of diagnostic."
        prompt_list = [zero_shot_text_prompt.replace("{label}", label) for label in categories]
        tokens = self.tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', max_length=100)
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        text_representation = self.text_encoder(input_ids, attention_mask)
        class_text_representation = {
            label: feature for label, feature in zip(categories, text_representation)
        }
        class_text_rep_tensor = torch.stack(list(class_text_representation.values()))

        return class_text_rep_tensor

    def similarity_classify(self, fea_concat) -> torch.Tensor:
        """
        ...
        """
        # Define categories
        categories = ["Normal beat",
                      "Supraventricular premature or ectopic beat (atrial or nodal)",
                      "Premature ventricular contraction",
                      "Fusion of ventricular and normal beat",
                      "Unclassifiable beat"]

        # Get text embeddings from Language Model
        text_embeddings = self.zero_shot_process_text(categories)

        # Calculate cosine similarity between the concatenated features and the text representation
        similarities = [F.cosine_similarity(fea.unsqueeze(0), text_embeddings) for fea in fea_concat]
        similarities = torch.stack(similarities).to(device)

        probabilities = F.softmax(similarities, dim=1).cpu().numpy()
        max_probability_class = np.argmax(probabilities, axis=1)
        max_probability_class = torch.tensor(max_probability_class).long()

        return max_probability_class

    def forward(self, fea_concat: torch.Tensor) -> torch.Tensor:
        """
        ...
        """
        pred = self.similarity_classify(fea_concat)

        return pred
