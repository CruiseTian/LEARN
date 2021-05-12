## Introduction

复现论文**LEARN Codes: Inventing Low-latency Codes via Recurrent Neural Networks**，并加以改进

## How To Run

```bash
git clone https://github.com/CruiseTian/LEARN.git && cd LEARN

# 设置环境(以下两种选择一种即可)
conda env create -f environment.yaml
# or 
conda create -n pytorch python=3.8.3   # 创建新的虚拟环境
source activate pytorch      # 激活新建的虚拟环境
pip install -r requirements.txt  # 安装对应依赖

# 执行主程序
python main.py -num_epoch 120
```

`main.py`运行还可以更改其他参数，具体详见`get_args.py`。

## File structure

logs -- log文件夹

tmp -- 模型文件夹

data -- 数据文件夹

## Plan
- [x] 增加attention机制