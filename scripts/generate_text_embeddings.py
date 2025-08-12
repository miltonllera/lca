import argparse
import h5py

import torch
from transformers import AutoTokenizer, AutoModel


def main(dataset_path, text_embedding_model, use_word_level_embeddings=False):
    torch.set_grad_enabled(False)

    with h5py.File(dataset_path, "r") as f:
        descriptions = [d.item().decode("utf8") for d in f["descriptions"]]  # type: ignore

    # load model
    model = AutoModel.from_pretrained(text_embedding_model)
    tokenizer = AutoTokenizer.from_pretrained(text_embedding_model)

    encoding = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
    token_embeddings = model.text_model(
        encoding['input_ids'], output_hidden_states=True
    ).last_hidden_state.cpu().numpy()

    if use_word_level_embeddings:
        raise NotImplementedError()

    with h5py.File(dataset_path, 'r+') as f:
        if 'token_embeddings' in f:
            del f['token_embeddings']
        dataset = f.create_dataset('token_embeddings', data=token_embeddings)
        dataset.attrs.create('text_embedding_model', text_embedding_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text embeddings from a dataset.")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset file (HDF5 format).",
    )

    parser.add_argument(
        "--text_embedding_model",
        type=str,
        required=True,
        help="Embedding model to use from HuggingFace"
    )

    main(**vars(parser.parse_args()))

