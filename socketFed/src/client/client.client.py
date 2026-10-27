# 打印差分隐私信息
if privacy_config.get('enabled', False):
    epsilon = defense_config.get('differential_privacy', {}).get('epsilon', 'N/A')
    delta = defense_config.get('differential_privacy', {}).get('delta', 'N/A')
    log_message = f"客户端 {self.client_id}: 应用差分隐私，消耗隐私预算 - Epsilon: {epsilon}, Delta: {delta}"
    print(log_message)
    self.logger.info(log_message)

# 打印量化、压缩、裁剪信息 (假设这些操作在内部完成或在此处模拟)
if compression_config.get('enabled', False):
    log_message = f"客户端 {self.client_id}: 应用模型量化压缩 (位数: {compression_config.get('quantization_bits', 'N/A')})"
    print(log_message)
    self.logger.info(log_message)

# 如果有裁剪操作，也在此处打印（目前代码中没有显式裁剪，这里仅作为示例）
if privacy_config.get('enabled', False) and privacy_config.get('clip_threshold') is not None:
    log_message = f"客户端 {self.client_id}: 应用模型梯度裁剪 (阈值: {privacy_config.get('clip_threshold')})"
    print(log_message)
    self.logger.info(log_message) 