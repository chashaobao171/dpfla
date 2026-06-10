import copy


# Get average weights
def average_weights(w, marks, float16_floats: bool = False):
    """
    Returns the average of the weights.

    Args:
        w: list of state_dict-like weight dicts.
        marks: list of client weights (float).
        float16_floats: if True, only floating-point tensors are cast to float16
            before aggregation (to mimic third-party float16 FedAvg behavior).
    """
    # NOTE:
    # - 对于浮点权重/缓冲区（float/half/bfloat16）可以做加权平均
    # - 对于非浮点缓冲区（如 BatchNorm 的 num_batches_tracked: LongTensor），
    #   做平均会导致 dtype 变化/加载失败/状态异常，尤其是 YOLO 这类模型包含大量 BN。
    #   这类 key 直接取第一个客户端的值（或保持不变）更安全。
    import torch

    if len(w) == 0:
        return {}

    marks_sum = float(sum(marks)) if sum(marks) != 0 else 1.0
    w_avg = copy.deepcopy(w[0])

    for key, v0 in w_avg.items():
        # 非Tensor（极少见）：直接取第一个
        if not torch.is_tensor(v0):
            w_avg[key] = v0
            continue

        # 非浮点Tensor：直接取第一个（避免 long/bool 被乘 float）
        if not v0.is_floating_point():
            w_avg[key] = v0
            continue

        # 浮点Tensor：做加权平均
        if float16_floats:
            # Third-party参考：对 float16 后再 mean/加权求和（仅浮点Tensor）
            # 注意：这里不动非浮点buffer（如 BN 的计数），避免 dtype/语义风险。
            acc = (v0.to(torch.float16) * marks[0]).to(torch.float16)
            for i in range(1, len(w)):
                acc = acc + (w[i][key].to(torch.float16) * marks[i])
            w_avg[key] = (acc * (torch.tensor(1.0 / marks_sum, device=acc.device, dtype=acc.dtype)))
        else:
            acc = v0 * marks[0]
            for i in range(1, len(w)):
                acc = acc + (w[i][key] * marks[i])
            w_avg[key] = acc * (1.0 / marks_sum)

    return w_avg
