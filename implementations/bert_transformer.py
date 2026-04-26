"""Run BERT transformer experiments for Energy Type and Risk classification."""

import pandas as pd
import torch
import torch.optim as optim

from modules.data_loader.bert_loader import df_to_bert_dataloader
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_tokenizer import BertTokenizerWrapper
from modules.embedding.bert_embedding import BertEmbeddingBackend
from modules.encoding.label_encoder import LabelEncoder
from modules.models.bert_classifier import BertClassifier
from modules.training_loop.config import _build_train_config
from modules.training_loop.train_loop import train_model_loop
from modules.training_loop.evaluate import evaluate


def encode_label_column(train_df, valid_df, test_df, label_col):
    """
    Fit label encoder on train labels and transform train/valid/test.
    
    :param train_df: Training dataframe containing the label column.
    :type train_df: pandas.DataFrame
    :param valid_df: Validation dataframe containing the label column.
    :type valid_df: pandas.DataFrame
    :param test_df: Test dataframe containing the label column.
    :type test_df: pandas.DataFrame
    :param label_col: Name of the column containing label data.
    :type label_col: str
    
    :returns: Tuple of (train_df, valid_df, test_df, label_encoder, class_names)
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, LabelEncoder, list[str]]
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[label_col].astype(str).tolist())

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df[label_col] = label_encoder.encode_many(train_df[label_col].astype(str).tolist())
    valid_df[label_col] = label_encoder.encode_many(valid_df[label_col].astype(str).tolist())
    test_df[label_col] = label_encoder.encode_many(test_df[label_col].astype(str).tolist())

    class_names = [
        label_encoder.id_to_label[i]
        for i in range(label_encoder.num_classes)
    ]

    return train_df, valid_df, test_df, label_encoder, class_names


def run_bert_experiment(
    train_df,
    valid_df,
    test_df,
    text_col,
    label_col,
    run_name,
    fine_tune=False,
    pooling="mean",
    batch_size=8,
    epochs=5,
):
    """
    Train and evaluate a BERT classifier for one target label.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, valid_df, test_df, label_encoder, class_names = encode_label_column(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_col=label_col,
    )

    bert_config = BertEmbeddingConfig(
        model_name="bert-base-uncased",
        max_length=160,
        dropout=0.1,
        pooling=pooling,
        fine_tune=fine_tune,
    )

    tokenizer_wrapper = BertTokenizerWrapper(bert_config)

    train_dl = df_to_bert_dataloader(
        train_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_dl = df_to_bert_dataloader(
        valid_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dl = df_to_bert_dataloader(
        test_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=False,
    )

    embedding_backend = BertEmbeddingBackend(bert_config)

    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=label_encoder.num_classes,
        dropout=0.2,
    ).to(device)

    optimiser = optim.AdamW(
        model.parameters(),
        lr=2e-5 if fine_tune else 1e-3,
    )

    config = _build_train_config(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        epochs=epochs,
        device=device,
        patience=2,
        criterion_weights=None,
        model_type="BERT",
        optimiser=optimiser,
        scheduler=None,
        need_length=False,
        energy_model=True,
        best_metric="f1_macro",
        save=True,
        num_classes=label_encoder.num_classes,
        run_name=run_name,
        extra_config={
            "test_dl": test_dl,
            "class_names": class_names,
            "threshold": 0.80,
            "use_temperature": False,
            "label_col": label_col,
            "pooling": pooling,
            "fine_tune": fine_tune,
        },
    )

    run_summary = train_model_loop(config)
    test_metrics = evaluate(config)

    return run_summary, test_metrics


# if __name__ == "__main__":
#     TEXT_COL = "Detailed Description of Event"

#     _, energy_metrics = run_bert_experiment(
#         train_path="dataset/model1_train.csv",
#         valid_path="dataset/model1_valid.csv",
#         test_path="dataset/model1_test.csv",
#         text_col=TEXT_COL,
#         label_col="Energy Type",
#         run_name="bert_energy_frozen_mean",
#         fine_tune=False,
#         pooling="mean",
#         batch_size=8,
#         epochs=5,
#     )

#     print("BERT Energy Type metrics:")
#     print(energy_metrics)

#     _, risk_metrics = run_bert_experiment(
#         train_path="dataset/model2_train.csv",
#         valid_path="dataset/model2_valid.csv",
#         test_path="dataset/model2_test.csv",
#         text_col=TEXT_COL,
#         label_col="Type of Potential Damage",
#         run_name="bert_risk_frozen_mean",
#         fine_tune=False,
#         pooling="mean",
#         batch_size=8,
#         epochs=5,
#     )

#     print("BERT Risk metrics:")
#     print(risk_metrics)