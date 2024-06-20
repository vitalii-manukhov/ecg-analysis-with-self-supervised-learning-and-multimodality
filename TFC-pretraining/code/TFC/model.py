"""
...
"""
from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

BERT_PRETRAIN_PATH = "../../BERT_pretrain/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TFC(nn.Module):
    """
    ...
    """
    def __init__(self, configs):
        super(TFC, self).__init__()

        self.adaptive_avgpool = nn.AdaptiveAvgPool1d(configs.TSlength_aligned)

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
        # Adaptive average pooling
        x_in_t = self.adaptive_avgpool(x_in_t)
        x_in_f = self.adaptive_avgpool(x_in_f)

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
        self.dimension_reducer = nn.Linear(768, 256)

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
        reduced_representation = self.dimension_reducer(sentence_representation)
        return reduced_representation


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
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT',
                                                       cache_dir=BERT_PRETRAIN_PATH)

    @staticmethod
    def get_diagnostic_string(label: int):
        class_names = {
            0: "Normal ECG",  # "Normal beat"
            1: "Myocardial Infarction",  # "Supraventricular premature beat"
            2: "ST/T change",  # "Premature ventricular contraction"
            3: "Hypertrophy",  # "Fusion of ventricular and normal beat"
            4: "Conducion Disturbance"  # "Unclassifiable beat"
        }

        if label in class_names:
            diagnostic_type = class_names[label]
            return f"The ECG of {diagnostic_type}, a type of diagnostic"
        else:
            return "Invalid label"

    def zero_shot_process_text(self, labels) -> torch.Tensor:
        """
        Description:
            Process the text data for zero-shot learning.

        Args:
            text_data: The text data to be processed.

        Returns:
            torch.Tensor: The processed text data.
        """
        categories = [
            "Normal ECG",
            "Myocardial Infarction",
            "ST/T change",
            "Hypertrophy",
            "Conducion Disturbance"
        ]

        prompts = [self.get_diagnostic_string(label.item()) for label in labels]
        tokens = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt', max_length=100)

        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        text_representation = self.text_encoder(input_ids, attention_mask)

        class_text_representation = {
            label: feature for label, feature in zip(categories, text_representation)
        }

        class_text_rep_tensor = torch.stack(list(class_text_representation.values()))

        return class_text_rep_tensor

    def similarity_classify(self, fea_concat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        ...
        """
        # Get text embeddings from Language Model
        class_text_rep_tensor = self.zero_shot_process_text(labels)

        # Calculate cosine similarity between the concatenated features and the text representation
        similarities = [F.cosine_similarity(elem.unsqueeze(0), class_text_rep_tensor) for elem in fea_concat]
        similarities = torch.stack(similarities)

        # probabilities = F.softmax(similarities, dim=1).cpu().detach().numpy()
        # max_probability_class = np.argmax(probabilities, axis=1)
        # max_probability_class = torch.tensor(max_probability_class).long()

        # return max_probability_class

        return similarities.to(device)

    def forward(self, fea_concat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        ...
        """
        pred = self.similarity_classify(fea_concat, labels)
        return pred
