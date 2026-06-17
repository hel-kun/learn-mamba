# Learn-Mamba

## Overview
私がMambaを学習するために作成したリポジトリですが、せっかくなのでpublicにすることにしました。
半分くらい(主要モデルなど)は論文を参考に自力で実装し、半分くらい(trainコードやテストコードなど)はCodexを活用しています。

自分だけで実装する前提なのでブランチを分けることもなく、~~mainに直プッシュしています。ご了承ください。~~ 結局ブランチを切ることにしました。

Paper：https://arxiv.org/abs/2312.00752

## Getting Started
あらかじめuv pythonを自身の環境にインストールしておいてください。
```bash
uv sync
```
で必要なライブラリがインストールされるはずです。

### Dataset
[SimpleStories_JP](https://huggingface.co/datasets/SimpleStories/SimpleStories-JA)を用いるようにしています。

### How to Train
```bash
uv run train.py --config <config_path>
```
で学習が始まります。Configについては、`configs/`以下のyamlファイルか、`config.py`を参考にしてください。

### How to Inference
```bash
uv run infer.py --checkpoint <checkpoint_path> --prompt <prompt_text> --max-new-tokens <max_new_tokens>
```
で動くはずです。
