"""
...
"""
import torch
import torch.nn.functional as F

BERT_PRETRAIN_PATH = "METS/BERT_pretrain/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_shot_precess_text(self, categories) -> torch.Tensor:
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
    self.class_text_representation = {
        label: feature for label, feature in zip(categories, text_representation)
    }


def similarity_classify(self, ecg_representation) -> torch.Tensor:
    """
    Description:
        Calculate the similarity between the ECG representation and the text representation.

    Args:
        ecg_representation (torch.Tensor): The ECG representation.

    Returns:
        torch.Tensor: The similarity between the ECG representation and the text representation.
    """
    class_text_rep_tensor = torch.stack(list(self.class_text_representation.values()))

    # Calculate cosine similarity between the ECG representation and the text representation
    similarities = [F.cosine_similarity(elem.unsqueeze(0), class_text_rep_tensor) for elem in ecg_representation]
    similarities = torch.stack(similarities).to(device)

    probabilities = F.softmax(similarities, dim=1).cpu().numpy()
    max_probability_class = np.argmax(probabilities, axis=1)
    # max_probabilities = np.max(probabilities, axis=1)

    max_probability_class = torch.tensor(max_probability_class).long()

    return max_probability_class
