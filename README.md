# TicTacToe with AOAT-MCTS

本项目实现了一个基于 **神经网络 + 蒙特卡洛树搜索（MCTS）** 的井字棋（Tic-Tac-Toe）智能体，并引入了 **AOAT（Adaptive Optimistic Action Truncation）节点选择策略** 来提升模型的决策效率和博弈性能。

项目支持：

* **人机对战**（play_human_vs_ai.py）
* **AI 自我博弈训练**（train.sh / train.bat）
* **模型对战评估**（Arena.py / Pit.py）
* **训练与模拟流程**（Learn.py / Simulate.py / Trainer.py）

---

## 📂 项目结构

```
tictactoe_with_AOAT-MCTS
├─ Arena.py                  # 两个AI对战的控制器
├─ Config.py                 # 配置文件（训练超参数等）
├─ go/                       # 神经网络与棋盘逻辑
│  ├─ GoGame.py              # 井字棋游戏逻辑（棋盘规则）
│  ├─ GoNNet.py              # 神经网络封装
│  ├─ GoPlayers.py           # 玩家类（AI/人类）
│  ├─ NNet.py                # PyTorch模型定义
│  └─ testCases.py           # 单元测试样例
├─ Learn.py                  # 学习流程（训练循环）
├─ MCTS.py                   # 蒙特卡洛树搜索实现
├─ Pit.py                    # 不同玩家或模型之间的对战
├─ pit_0.txt                 # 对战结果日志
├─ play_human_vs_ai.py       # 人类玩家 vs AI 对战入口
├─ Simulate.py               # 自我博弈模拟器
├─ test.ipynb                # Jupyter Notebook 测试
├─ train.bat                 # Windows 一键训练脚本
├─ train.sh                  # Linux/Mac 一键训练脚本
├─ Trainer.py                # 训练器核心逻辑
└─ utils/                    # 工具函数与日志模块
   ├─ eval.py
   ├─ logger.py
   ├─ misc.py
   ├─ progress/              # 进度条库
   │  ├─ demo.gif
   │  ├─ progress/bar.py
   │  ├─ progress/counter.py
   │  ├─ progress/spinner.py
   │  └─ ...
   ├─ utils.py
   └─ __init__.py
```

---

## 🚀 使用方法

### 1. 环境配置

```bash
conda create -n tictactoe python=3.9 -y
conda activate tictactoe
pip install torch numpy tqdm
```

（可选）如需可视化进度条：

```bash
pip install -e utils/progress
```

---

### 2. 训练AI

#### Windows

```bash
train.bat
```

#### Linux / Mac

```bash
bash train.sh
```

训练过程会生成模型权重文件（如 `checkpoint.pth.tar`）。

---

### 3. 人机对战

运行以下命令进入井字棋对战：

```bash
python play_human_vs_ai.py
```

程序会让你选择落子位置，AI 会用 **AOAT-MCTS + 神经网络策略** 决策下一步。
支持先后手选择。

---

### 4. AI 对战

对比两个AI的实力：

```bash
python Pit.py
```

你可以修改 `Pit.py` 选择不同的模型进行对战测试。

---

### 5. 继续训练

若已有模型，可以继续从上次权重开始训练：

```bash
python Learn.py
```

---

## 📑 主要文件说明

* **Arena.py**：管理 AI 对战过程，统计胜负结果。
* **MCTS.py**：实现了蒙特卡洛树搜索，并结合 AOAT 策略优化。
* **GoGame.py**：井字棋游戏环境，定义棋盘、规则与合法动作。
* **GoNNet.py / NNet.py**：基于 PyTorch 的神经网络模型，策略与价值输出。
* **Trainer.py**：核心训练器，包括自对弈、数据收集和网络更新。
* **play_human_vs_ai.py**：人类玩家与 AI 交互入口。

---
