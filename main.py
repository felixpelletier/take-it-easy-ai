from tqdm import tqdm
import torch
import torch.utils.tensorboard
from torch import nn
import random
import time
from collections import deque
from itertools import chain
import math
from statistics import median
import glob
import datetime
import os, sys
import gc
import gzip
from dataclasses import dataclass
import cProfile

from checkpoint_helpers import get_latest_checkpoint_path

IS_DEBUG = False
OUT_DIR = "out"

LEARNING_RATE = 1e-5
LOW_LEARNING_RATE = 1e-6
RESTART_PERIOD = 10000

DISCOUNT_RATE = 0.98

EPOCHS = 5_000_000
STEPS = 100
E_START = 0.01
E_STOP = 0.0
E_PERIOD = 500

MEMORY_SIZE = 1_000_000
SAMPLE_SIZE = 1024
MINIBATCH_SIZE = 128
SELECTIVE_MEMORY_ENABLED = False
IMPROVEMENT_ENRICHMENT = False
ANNEALING_STEPS = 1000

HIGH_SCORE_POWER = 1.0

WIDTH = 768
GRADIENT_CLIPPING_NORM_THRESHOLD = 10
GRADIENT_CLIPPING_VALUE_THRESHOLD = 100

def create_board():
    return (None, ) * 19

def create_tileset():
    result = []
    for i in [1, 5, 9]:
        for j in [2, 6, 7]:
            for k in [3, 4, 8]:
                result.append((i, j, k))
    return tuple(result)

def get_average_board_score(board, tiles):
    empty_tiles_positions = [i for i, placed_tile in enumerate(board) if placed_tile is None]

    if not empty_tiles_positions:
        return get_board_score(board)

    scores = []
    for tile_index in range(len(tiles)):
        max_score_for_tile = 0
        for empty_tile_position in empty_tiles_positions:
            new_board = list(board)
            new_board[empty_tile_position] = tiles[tile_index]
            new_board = tuple(new_board)

            new_tiles = list(tiles)
            del new_tiles[tile_index]
            new_tiles = tuple(new_tiles)
            
            max_score_for_tile = max(max_score_for_tile, get_average_board_score(new_board, new_tiles))
        scores.append(max_score_for_tile)

    return float(sum(scores)) / len(scores)


def improve_board(current_board, annealing_steps):
    current_board_score = get_board_score(current_board[:19])

    best_board_yet = list(current_board)
    best_score_yet = current_board_score

    positions = list(range(len(best_board_yet)))

    for j in range(annealing_steps):
        temperature = 1000 * (1 - ((j+1) / annealing_steps)) + 1

        n1, n2 = random.sample(positions, 2)

        if type(current_board) != list and type(current_board) != tuple:
            print(current_board)
        new_board = list(current_board)
        new_board[n1], new_board[n2] = new_board[n2], new_board[n1]

        score = get_board_score(new_board[:19])
        p = math.exp(-(current_board_score - score)/temperature)

        if score > current_board_score or random.random() < p:
            current_board = new_board
            current_board_score = score

        if current_board_score > best_score_yet:
            best_score_yet = current_board_score
            best_board_yet = list(current_board)

    return best_board_yet, best_score_yet


def get_board_score(board):
    score = 0
    if board[3] is not None and board[8] is not None and board[13] is not None and board[3][0] == board[8][0] and board[8][0] == board[13][0]:
        score += 3 * board[3][0]
    if board[5] is not None and board[10] is not None and board[15] is not None and board[5][0] == board[10][0] and board[10][0] == board[15][0]:
        score += 3 * board[5][0]
    if board[1] is not None and board[6] is not None and board[11] is not None and board[16] is not None and board[1][0] == board[6][0] and board[1][0] == board[11][0] and board[1][0] == board[16][0]:
        score += 4 * board[1][0]
    if board[2] is not None and board[7] is not None and board[12] is not None and board[17] is not None and board[2][0] == board[7][0] and board[2][0] == board[12][0] and board[2][0] == board[17][0]:
        score += 4 * board[2][0]
    if board[0] is not None and board[4] is not None and board[9] is not None and board[14] is not None and board[18] is not None and board[0][0] == board[4][0] and board[0][0] == board[9][0] and board[0][0] == board[14][0] and board[0][0] == board[18][0]:
        score += 5 * board[0][0]

    if board[0] is not None and board[1] is not None and board[3] is not None and board[0][1] == board[1][1] and board[0][1] == board[3][1]:
        score += 3 * board[0][1]
    if board[15] is not None and board[17] is not None and board[18] is not None and board[15][1] == board[17][1] and board[15][1] == board[18][1]:
        score += 3 * board[15][1]
    if board[10] is not None and board[12] is not None and board[14] is not None and board[16] is not None and board[10][1] == board[12][1] and board[10][1] == board[14][1] and board[10][1] == board[16][1]:
        score += 4 * board[10][1]
    if board[2] is not None and board[4] is not None and board[6] is not None and board[8] is not None and board[2][1] == board[4][1] and board[2][1] == board[6][1] and board[2][1] == board[8][1]:
        score += 4 * board[2][1]
    if board[5] is not None and board[7] is not None and board[9] is not None and board[11] is not None and board[13] is not None and board[5][1] == board[7][1] and board[5][1] == board[9][1] and board[5][1] == board[11][1] and board[5][1] == board[13][1]:
        score += 5 * board[5][1]

    if board[0] is not None and board[2] is not None and board[5] is not None and board[0][2] == board[2][2] and board[0][2] == board[5][2]:
        score += 3 * board[0][2]
    if board[13] is not None and board[16] is not None and board[18] is not None and board[13][2] == board[16][2] and board[13][2] == board[18][2]:
        score += 3 * board[13][2]
    if board[1] is not None and board[4] is not None and board[7] is not None and board[10] is not None and board[1][2] == board[4][2] and board[1][2] == board[7][2] and board[1][2] == board[10][2]:
        score += 4 * board[1][2]
    if board[8] is not None and board[11] is not None and board[14] is not None and board[17] is not None and board[8][2] == board[11][2] and board[8][2] == board[14][2] and board[8][2] == board[17][2]:
        score += 4 * board[8][2]
    if board[3] is not None and board[6] is not None and board[9] is not None and board[12] is not None and board[15] is not None and board[3][2] == board[6][2] and board[3][2] == board[9][2] and board[3][2] == board[12][2] and board[3][2] == board[15][2]:
        score += 5 * board[3][2]

    return score

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def convert_board_to_tuple_of_tuples(board, tileset):
    def generator():
        for t in board:
            if t is None:
                yield (0, 0, 0)
            else:
                yield tileset[t]

    return tuple(generator())


def state_to_tensor(board: list[int], tileset: list[tuple[int]], next_tile_index: int):
    return torch.tensor(tuple(chain(*(convert_board_to_tuple_of_tuples(board, tileset) + (tileset[next_tile_index],)))), dtype=torch.float)


class BasicBlock(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.stack = nn.Sequential(
            nn.BatchNorm1d(in_width),
            nn.GELU(),
            nn.Linear(in_width, out_width),
        )

    def forward(self, x):
        return self.stack(x)

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stack = nn.Sequential(
            BasicBlock(width, width),
            BasicBlock(width, width),
            BasicBlock(width, width),
        )

        self.stack.apply(init_weights) # Not sure it's necessary since the one in the parent might do the job. Better safe than sorry.

    def forward(self, x):
        residual = x
        out = self.stack(x)
        out += residual
        return out


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.main_stack = nn.Sequential(
            nn.Linear((19 + 1) * 3, WIDTH),

            ResidualBlock(WIDTH),
            ResidualBlock(WIDTH),

            BasicBlock(WIDTH, 19),
        )

        self.main_stack.apply(init_weights)

    def forward(self, x):
        score = self.main_stack(x)
        return score


class Nop(object):
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop

@dataclass
class Move:
    prev_board: list
    state: torch.tensor
    action: int
    new_score: int

@dataclass
class MemoryCell:
    state: torch.tensor
    action: int
    reward: float

torch.serialization.add_safe_globals([MemoryCell])

class Game:
    def __init__(self, tileset, device):
        self.board = list(create_board())
        self.tileset_indices = list(range(len(tileset)))
        self.tileset = tileset
        self.device = device
        random.shuffle(self.tileset_indices)

        self.random_move_occured = False

        self.moves = []

    def is_done(self):
        return all(tile is not None for tile in self.board)

    def get_state(self):
        next_tile_index = self.tileset_indices[-1]
        return state_to_tensor(self.board, self.tileset, next_tile_index).to(device=self.device)

    def get_score(self):
        return get_board_score(tuple(self.tileset[i] if i else None for i in self.board))

    def get_next_tile(self):
        if self.tileset_indices:
            return self.tileset_indices[-1]
        else:
            return None

    def apply_action(self, action):
        prev_board = list(self.board)
        state = self.get_state()
        is_legal = self.board[action] is None
        if is_legal:
            self.board[action] = self.tileset_indices.pop()

        self.moves.append(Move(
            prev_board=prev_board,
            state=state,
            action=action,
            new_score=get_board_score(tuple(self.tileset[i] if i else None for i in self.board))
        ))


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert device == "cuda"

    start_timestamp = str(int(time.time()))

    model = NeuralNetwork().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=1e-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=RESTART_PERIOD, T_mult=1, eta_min=LOW_LEARNING_RATE)
    loss_fn = nn.MSELoss()
    moves_memory = deque(maxlen=MEMORY_SIZE)

    starting_epoch = 0
    should_resume = 'Y' in input("Resume? (Y/N): ").strip().upper()
    if should_resume:
        checkpoint_path = get_latest_checkpoint_path(OUT_DIR)
        print(f"Resuming from '{checkpoint_path}'")

        with gzip.GzipFile(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        starting_epoch = checkpoint['epoch'] + 1
        start_timestamp = checkpoint['start_timestamp']

        if 'moves_memory' in checkpoint:
            moves_memory.extend(checkpoint['moves_memory'])

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "runs", f"{start_timestamp}_{current_time}_{starting_epoch}"
    )

    if IS_DEBUG:
        print("DEBUGGING! Not initializing writer.")
        writer = Nop()
    else:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir, flush_secs=30, max_queue=100)

    tileset = create_tileset()
    score_smoother = deque(maxlen=100)
    pr = cProfile.Profile()
    for epoch in tqdm(range(starting_epoch, EPOCHS), initial=starting_epoch, total=EPOCHS):
        if epoch == 200:
            pr.enable()
        eps = ((EPOCHS - epoch) / EPOCHS) * (E_START - E_STOP) * (0.5 * math.cos((epoch / float(E_PERIOD)) * (2 * math.pi)) + 0.5) + E_STOP

        epoch_scores = []
        epoch_scores_without_exploration = []
        epoch_scores_enhancements_only = []

        model.eval()
        with torch.no_grad():
            games = [Game(tileset, device) for step_number in range(STEPS)]
            while not games[0].is_done():
                predictions = model(torch.stack([game.get_state() for game in games]))

                for game_number, game in enumerate(games):
                    if random.random() < eps:
                        game.random_move_occured = True
                        action = random.choice([i for i, tile in enumerate(game.board) if tile is None])
                    else:
                        prediction = predictions[game_number]
                        sorted_actions = prediction.argsort(0).flip(0)
                        action = None
                        for possible_action in sorted_actions:
                            action = possible_action.item()
                            is_legal = game.board[action] is None
                            if is_legal:
                                break

                    game.apply_action(action)

            for game in games:
                score = get_board_score(tuple(game.tileset[i] for i in game.board))
                epoch_scores.append(score)
                if not game.random_move_occured:
                    epoch_scores_without_exploration.append(score)
                moves = list(game.moves)

                if not SELECTIVE_MEMORY_ENABLED or (score / 307) > random.random() or random.random() < 0.05:
                    for j, move in enumerate(moves):
                        final_reward = score * (DISCOUNT_RATE ** j)
                        current_score = move.new_score
                        reward = max(current_score, final_reward)

                        moves_memory.append(MemoryCell(
                            state=move.state,
                            action=move.action,
                            reward=reward
                        ))

                if IMPROVEMENT_ENRICHMENT and ANNEALING_STEPS > 0:
                    improved_board, improved_score = improve_board(convert_board_to_tuple_of_tuples(game.board, game.tileset), ANNEALING_STEPS)
                    epoch_scores_enhancements_only.append(improved_score)
                    if improved_score > score:
                        board = list(create_board())
                        indices = list(range(len(improved_board)))
                        random.shuffle(indices)
                        for j, tile_index in enumerate(indices):
                            prev_board = list(board)

                            tuple_state = tuple(t if t is not None else (0, 0, 0) for t in prev_board) + (improved_board[tile_index],)
                            state = torch.tensor(tuple(chain(*tuple_state)), dtype=torch.float).to(device=device)
                            board[tile_index] = improved_board[tile_index]

                            final_reward = improved_score * (DISCOUNT_RATE ** (len(board) - j - 1))
                            current_score = get_board_score(board)
                            reward = max(current_score, final_reward)

                            moves_memory.append(MemoryCell(
                                state=state,
                                action=tile_index,
                                reward=reward
                            ))

            s_optimize = time.time()
            epoch_loss = []
            batch = random.sample(moves_memory, min(len(moves_memory), SAMPLE_SIZE))

        if epoch == 200:
            print(time.time() - s_optimize)
            pr.disable()
            pr.dump_stats(f"profile_{time.time()}")

        model.train()
        for minibatch in batched(batch, MINIBATCH_SIZE):
            optimizer.zero_grad()

            states = []
            actions = []
            rewards = []

            for move in minibatch:
                states.append(move.state)
                actions.append(torch.tensor(move.action).to(device))
                rewards.append(torch.tensor(move.reward).to(device))

            state_batch = torch.stack(states)
            action_batch = torch.stack(actions)
            reward_batch = torch.stack(rewards)

            prediction = model(state_batch)
            prediction_batch = prediction.gather(1, action_batch.unsqueeze(1)).squeeze()

            loss = loss_fn(prediction_batch, reward_batch)
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(model.parameters(), GRADIENT_CLIPPING_VALUE_THRESHOLD)
            optimizer.step()

        median_score_smooth = None
        if len(epoch_scores_without_exploration):
            score_smoother.append(median(epoch_scores_without_exploration))
            median_score_smooth = float(sum(score_smoother)) / len(score_smoother)
        scheduler.step()

        print(f"Avg score: {float(sum(epoch_scores)) / len(epoch_scores)}")
        writer.add_scalar("Avg score", float(sum(epoch_scores)) / len(epoch_scores), epoch)
        print(f"Median score: {median(epoch_scores)}")
        writer.add_scalar("Median score", median(epoch_scores), epoch)
        print(f"Min score: {min(epoch_scores)}")
        writer.add_scalar("Min score", min(epoch_scores), epoch)
        print(f"Max score: {max(epoch_scores)}")
        writer.add_scalar("Max score", max(epoch_scores), epoch)
        print()
        if len(epoch_scores_without_exploration) > 0:
            print(f"Avg score without exploration: {float(sum(epoch_scores_without_exploration)) / len(epoch_scores_without_exploration)}")
            writer.add_scalar("Avg score without exploration", float(sum(epoch_scores_without_exploration)) / len(epoch_scores_without_exploration), epoch)
            print(f"Median score without exploration: {median(epoch_scores_without_exploration)}")
            writer.add_scalar("Median score without explorration", median(epoch_scores_without_exploration), epoch)
            if median_score_smooth is not None:
                print(f"Median score without exploration (smooth): {median_score_smooth}")
                writer.add_scalar("Median score without exploration (smooth)", median_score_smooth, epoch)
            print(f"Min score without exploration: {min(epoch_scores_without_exploration)}")
            writer.add_scalar("Min score without exploration", min(epoch_scores_without_exploration), epoch)
            print(f"Max score without exploration: {max(epoch_scores_without_exploration)}")
            writer.add_scalar("Max score without exploration", max(epoch_scores_without_exploration), epoch)
            print()
        if len(epoch_scores_enhancements_only) > 0:
            print(f"Avg score of enhancements: {float(sum(epoch_scores_enhancements_only)) / len(epoch_scores_enhancements_only)}")
            writer.add_scalar("Avg score of enhancements", float(sum(epoch_scores_enhancements_only)) / len(epoch_scores_enhancements_only), epoch)
            print(f"Median score of enhancement: {median(epoch_scores_enhancements_only)}")
            writer.add_scalar("Median score of enhancements", median(epoch_scores_enhancements_only), epoch)
            print(f"Min score of enhancement: {min(epoch_scores_enhancements_only)}")
            writer.add_scalar("Min score of enhancements", min(epoch_scores_enhancements_only), epoch)
            print(f"Max score of enhancement: {max(epoch_scores_enhancements_only)}")
            writer.add_scalar("Max score of enhancements", max(epoch_scores_enhancements_only), epoch)
            print()
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch)
        print(f"epsilon: {eps}")
        writer.add_scalar("epsilon", eps, epoch)
        print(f"moves_memory size: {len(moves_memory)}")
        print()
        print(f"Avg loss: {float(sum(epoch_loss)) / len(epoch_loss)}")
        writer.add_scalar("Avg loss", float(sum(epoch_loss)) / len(epoch_loss), epoch)
        print(f"Median loss: {median(epoch_loss)}")
        writer.add_scalar("Median loss", median(epoch_loss), epoch)
        print(f"Optimize: {time.time() - s_optimize}s")

        if epoch % 5000 == 0 and epoch > 0 and epoch != starting_epoch:
            prefix = f"takeiteasy_{start_timestamp}"
            specific_outdir = os.path.join(OUT_DIR, prefix)
            os.makedirs(specific_outdir, exist_ok=True)

            with gzip.GzipFile(os.path.join(specific_outdir, f"{prefix}_{epoch}.pth.gz"), 'wb') as f:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'moves_memory': list(moves_memory),
                'start_timestamp': start_timestamp,
                }, f)

    writer.close()
