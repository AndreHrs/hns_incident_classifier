import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def df_to_dataloader(
    df,
    tokens_col: str,
    energy_col: str,
    risk_col: str,
    batch_size: int,
    pad_id: int = 0,
    shuffle: bool = True,
):
    """
    Convert a preprocessed DataFrame into a DataLoader compatible with _unpack_batch.

    Each batch yields: (D, DL, Energy, Risk)
        - D      : padded token-id tensor         (batch, max_seq_len)  long
        - DL     : original sequence lengths       (batch,)              long
        - Energy : energy type label               (batch,)              long
        - Risk   : potential damage label          (batch,)              long

    Args:
        df         : DataFrame output from the preprocessing pipeline.
        tokens_col : Column of token-id lists (e.g. "description_tokens").
        energy_col : Column of energy labels (e.g. "energy_type").
                     MUST already be integer-encoded — string/categorical values will
                     cause a RuntimeError when converting to a torch.long tensor.
        risk_col   : Column of risk labels (e.g. "potential_damage").
                     MUST already be integer-encoded — same requirement as energy_col.
        batch_size : Batch size.
        pad_id     : Value used to pad shorter sequences. Defaults to 0.
        shuffle    : Whether to shuffle the DataLoader.

    Returns:
        DataLoader yielding (D, DL, Energy, Risk) batches.
    """
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in df[tokens_col]]
    lengths   = torch.tensor([len(s) for s in sequences], dtype=torch.long)

    D = pad_sequence(sequences, batch_first=True, padding_value=pad_id)

    Energy = torch.tensor(df[energy_col].to_numpy(), dtype=torch.long)
    Risk   = torch.tensor(df[risk_col].to_numpy(),   dtype=torch.long)

    dataset = TensorDataset(D, lengths, Energy, Risk)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
