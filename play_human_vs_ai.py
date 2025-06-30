import torch
import random
import numpy as np
import time
from copy import deepcopy
from Config import get_config
from go.GoGame import GoGame
from go.NNet import NNetWrapper as nn
from MCTS import MCTS

class MCTSStrategy:
    def __init__(self, mcts):
        self.mcts = mcts

    def next_move(self, game, player):
        print("AI 正在思考中...")
        pi = self.mcts.getActionProb(deepcopy(game))
        action = np.argmax(pi)
        time.sleep(0.5)
        return action

class HumanStrategy:
    def __init__(self):
        pass

    def next_move(self, game, player):
        valids = game.getValidMoves()
        while True:
            try:
                pos = int(input("请输入你的落子位置（1-9）：")) - 1
                if pos < 0 or pos >= 9 or valids[pos] == 0:
                    print("无效位置，请重新输入！")
                else:
                    return pos
            except:
                print("输入错误，请输入数字1~9")

def action_to_xy(action):
    return divmod(action, 3)

def print_board(board):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print("\n  1 2 3\n -------")
    for i in range(3):
        row = f"{i+1}|"
        for j in range(3):
            row += symbols[int(board[i][j])] + " "
        print(row)
    print(" -------")

def play_human_vs_ai():
    IterNumber = 45
    seed = IterNumber
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_config(IterNumber).parse_args([])
    game = GoGame(3, "tic_tac_toe")
    nnet = nn(3, args)
    nnet.load_checkpoint('./.checkpoints/', 'AOAT-Bernoulli-Pi_45.pth.tar')
    # nnet.load_checkpoint('./.checkpoints/', 'AOAT-Bernoulli_45.pth.tar')
    # nnet.load_checkpoint('./.checkpoints/', 'UCT_45.pth.tar')
    mcts = MCTS(nnet, game, args)

    # 选择先后手
    while True:
        choice = input("请选择先手（输入 1 表示你先手，输入 2 表示AI先手）：")
        if choice == "1":
            human_player = 1
            break
        elif choice == "2":
            human_player = -1
            break
        else:
            print("输入无效，请输入 1 或 2")

    # 固定从 X（1）开始
    game.cur_player = 1
    print(f"\n你是 {'X' if human_player == 1 else 'O'}，AI 是 {'O' if human_player == 1 else 'X'}。开始游戏！")

    strategies = {
        human_player: HumanStrategy(),
        -human_player: MCTSStrategy(mcts)
    }

    print_board(game.board)

    while True:
        current_player = game.cur_player
        strategy = strategies[current_player]
        action = strategy.next_move(game, current_player)

        game.ExcuteAction(action)
        print_board(game.board)

        result = game.getGameEnded()
        if result != -1:
            if result == 0.5:
                print("平局！")
            elif result == 1:
                print("先手胜利！")
            elif result == 0:
                print("后手胜利！")
            else:
                print("未知游戏结果！")
            break
        else:
            print("游戏继续...")

if __name__ == '__main__':
    play_human_vs_ai()
