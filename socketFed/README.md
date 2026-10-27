# SocketFed - 基于Socket的联邦学习框架

这是一个基于Socket通信的联邦学习框架，支持分布式训练、模型聚合、防御机制和攻击模拟等功能。

## 项目结构

```
socketFed/
├── config/                 # 配置文件目录
├── src/                    # 源代码目录
│   ├── server/            # 服务器模块
│   ├── client/            # 客户端模块
│   ├── aggregation/       # 聚合算法模块
│   ├── defense/           # 防御机制模块
│   ├── attack/            # 攻击模块
│   └── utils/             # 工具函数
├── examples/              # 示例代码
├── tests/                 # 测试代码
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 配置服务器和客户端参数（config/config.yaml）
2. 启动服务器：
```bash
python src/server/server.py
```
3. 启动客户端：
```bash
python src/client/client.py
```

## 主要特性

- 基于Socket的分布式通信
- 支持多种聚合算法（FedAvg, FedProx等）
- 内置防御机制（差分隐私、异常检测等）
- 支持模拟攻击场景
- 可扩展的模块化设计

## 示例

查看 `examples/` 目录获取使用示例。 