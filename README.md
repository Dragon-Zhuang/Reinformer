# Rein*for*mer
Official code for ICML 2024 paper **Rein*for*mer**: Max-Return Sequence Modeling for offline RL

Here is the overview of our proposed **Rein*for*mer**. For more details, please refer to our paper https://arxiv.org/pdf/2405.08740.
![overview](https://github.com/Dragon-Zhuang/Reinformer/assets/47406013/698651f2-24c5-4734-9423-97088058bef7)

## Quick Start
1. Process the data.

`python data/download_d4rl_datasets.py`

2. Train the model.

`python main.py --env hopper --dataset medium`
