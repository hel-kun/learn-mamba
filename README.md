# Learn-Mamba

## Overview
私がMambaを学習するために作成したリポジトリですが、せっかくなのでpublicにすることにしました。
半分くらい(主要モデルなど)は論文を参考に自力で実装し、半分くらい(trainコードやテストコードなど)はCodexを活用しています。

自分だけで実装する前提なのでブランチを分けることもなく、mainに直プッシュしています。ご了承ください。

Paper：https://arxiv.org/abs/2312.00752

## Getting Started
あらかじめuv pythonを自身の環境にインストールしておいてください。
```bash
uv sync
```

### Dataset
[SimpleStories_JP](https://huggingface.co/datasets/SimpleStories/SimpleStories-JA)を用いるようにしているつもりです。

### How to Train
```bash
uv run train.py --config <config_path>
```
で動くようにするつもりです。
現状、コードは完成していないため、完成次第更新していきます。

### How to Inference
```bash
uv run infer.py --config <config_path>
```
で動くようにするつもりです。
現状、コードは完成していないため、完成次第更新していきます。
