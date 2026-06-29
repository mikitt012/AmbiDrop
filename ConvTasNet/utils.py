def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T]
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: list of numpy arrays, each [C, T] or [T] (T varies per sample)
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3:
            results.append(input[:, :length].view(C, -1).cpu().numpy())
        elif dim == 2:
            results.append(input[:length].view(-1).cpu().numpy())
    return results
