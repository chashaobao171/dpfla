def get_client_config(self):
    """获取客户端配置"""
    return {
        'model': 'simple_cnn',  # 使用 SimpleCNN 模型
        'model_config': {
            'simple_cnn': {
                'in_channels': 3,
                'num_classes': 10,
                'dropout_rate': 0.5
            }
        },
        'batch_size': 32,
        'epochs': 1,
        'max_epochs': 1,
        'optimizer': {
            'type': 'adam',
            'config': {
                'learning_rate': 0.001,
                'betas': [0.9, 0.999],
                'eps': '1e-8',
                'weight_decay': 0
            }
        }
    } 