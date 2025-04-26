from tqdm import tqdm
import torch
import torch.utils.tensorboard
from torch import nn
import numpy as np
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

DEBUG = False
DEBUG_RESUME = False

OUT_DIR = "out"

NUMBER_OF_TILES_IN_BOARD = 19
STATE_SIZE = (NUMBER_OF_TILES_IN_BOARD + 1) * 3

SEED = 42

LEARNING_RATE = 1e-5
LOW_LEARNING_RATE = 1e-6
RESTART_PERIOD = 1000

DISCOUNT_RATE = 0.98

EPOCHS = 1
PARALLEL_GAMES_COUNT = 32
MAX_STEPS_PER_GAME = NUMBER_OF_TILES_IN_BOARD
BATCH_SIZE = MAX_STEPS_PER_GAME * PARALLEL_GAMES_COUNT
MINIBATCH_SIZE = 32

ITERATIONS = 5_000_000

ACTOR_NETWORK_WIDTH = 1024
CRITIC_NETWORK_WIDTH = 1024

CLIP_COEF = 0.2
ENTROPY_COEF = 0.01

VALUE_GRADIENT_CLIPPING_VALUE_THRESHOLD = 100

def create_board():
    return (None, ) * NUMBER_OF_TILES_IN_BOARD

def create_tileset():
    result = []
    for i in [1, 5, 9]:
        for j in [2, 6, 7]:
            for k in [3, 4, 8]:
                result.append((i, j, k))
    return tuple(result)

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

def init_weights(std=np.sqrt(2)):
    def inner(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, std)
            m.bias.data.fill_(0.0)
    return inner


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
            # nn.Tanh(),
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


class PolicyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.main_stack = nn.Sequential(
            nn.Linear(STATE_SIZE, ACTOR_NETWORK_WIDTH),

            ResidualBlock(ACTOR_NETWORK_WIDTH),
            ResidualBlock(ACTOR_NETWORK_WIDTH),

            BasicBlock(ACTOR_NETWORK_WIDTH, 19),
        )

        self.main_stack.apply(init_weights(0.01))

    def forward(self, x):
        out = self.main_stack(x)
        out1 = nn.functional.softmax(out, dim=-1)

        mask = x[:, 0:NUMBER_OF_TILES_IN_BOARD*3:3] == 0

        out2 = out1 * mask
        out3 = nn.functional.normalize(out2, p=1, dim=-1)
        return out3

class ValueNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.main_stack = nn.Sequential(
            nn.Linear(STATE_SIZE, CRITIC_NETWORK_WIDTH),

            ResidualBlock(CRITIC_NETWORK_WIDTH),
            ResidualBlock(CRITIC_NETWORK_WIDTH),

            BasicBlock(CRITIC_NETWORK_WIDTH, 1),
        )

        self.main_stack.apply(init_weights())

    def forward(self, x):
        return self.main_stack(x)


class Nop(object):
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop

@dataclass
class Move:
    prev_board: list
    state: torch.tensor
    action: int
    new_score: int
    expected_reward: float

@dataclass
class MemoryCell:
    state: torch.tensor
    action: int
    reward: float
    expected_reward: float

torch.serialization.add_safe_globals([MemoryCell])

class Game:
    def __init__(self, tileset, device):
        self.board = list(create_board())
        self.tileset_indices = list(range(len(tileset)))
        self.tileset = tileset
        self.device = device
        self._is_done = False
        self._score = 0
        random.shuffle(self.tileset_indices)

        self.random_move_occured = False

        self.moves = []

    def is_done(self):
        if self._is_done:
            return True

        self._is_done = all(tile is not None for tile in self.board)
        return self._is_done

    def get_state(self):
        next_tile_index = self.tileset_indices[-1]
        return state_to_tensor(self.board, self.tileset, next_tile_index).to(device=self.device)

    def get_valid_actions_mask(self):
        return torch.tensor([t is None for t in self.board], device=self.device)

    def get_score(self):
        return self._score

    def _compute_score(self):
        return get_board_score(tuple(self.tileset[i] if i else None for i in self.board))

    def get_next_tile(self):
        if self.tileset_indices:
            return self.tileset_indices[-1]
        else:
            return None

    def apply_action(self, action):
        prev_board = list(self.board)
        is_legal = self.board[action] is None
        if is_legal:
            self.board[action] = self.tileset_indices.pop()
            self._score = self._compute_score()

        return is_legal


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert device == "cuda"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    start_timestamp = str(int(time.time()))

    policy_model = PolicyNeuralNetwork().to(device)
    policy_optimizer = torch.optim.AdamW(
        policy_model.parameters(), 
        lr=LEARNING_RATE, 
        amsgrad=True, 
        weight_decay=1e-1
    )
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(policy_optimizer, T_0=RESTART_PERIOD, T_mult=1, eta_min=LOW_LEARNING_RATE)

    value_model = ValueNeuralNetwork().to(device)
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=1e-1)
    value_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(value_optimizer, T_0=RESTART_PERIOD, T_mult=1, eta_min=LOW_LEARNING_RATE)
    value_loss_fn = nn.MSELoss()

    starting_iteration = 0
    should_resume = not (DEBUG and not DEBUG_RESUME) and 'Y' in input("Resume? (Y/N): ").strip().upper()
    if should_resume:
        checkpoint_path = get_latest_checkpoint_path(OUT_DIR)
        print(f"Resuming from '{checkpoint_path}'")

        with gzip.GzipFile(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, weights_only=True)

        policy_model.load_state_dict(checkpoint['policy']['model_state_dict'])
        policy_optimizer.load_state_dict(checkpoint['policy']['optimizer_state_dict'])
        policy_scheduler.load_state_dict(checkpoint['policy']['scheduler_state_dict'])
        policy_scheduler.base_lrs = [LEARNING_RATE for _ in policy_scheduler.base_lrs]
        policy_scheduler.eta_min = LOW_LEARNING_RATE
        policy_scheduler.T_0 = RESTART_PERIOD
        policy_scheduler.T_i = RESTART_PERIOD

        value_model.load_state_dict(checkpoint['value']['model_state_dict'])
        value_optimizer.load_state_dict(checkpoint['value']['optimizer_state_dict'])
        value_scheduler.load_state_dict(checkpoint['value']['scheduler_state_dict'])
        value_scheduler.base_lrs = [LEARNING_RATE for _ in value_scheduler.base_lrs]
        value_scheduler.eta_min = LOW_LEARNING_RATE
        value_scheduler.T_0 = RESTART_PERIOD
        value_scheduler.T_i = RESTART_PERIOD

        starting_iteration = checkpoint['iteration'] + 1
        start_timestamp = checkpoint['start_timestamp']

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "runs", f"{start_timestamp}_{current_time}_{starting_iteration}"
    )

    if DEBUG:
        print("DEBUGGING! Not initializing writer.")
        writer = Nop()
    else:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir, flush_secs=30, max_queue=100)

    tileset = create_tileset()
    score_smoother = deque(maxlen=100)
    pr = cProfile.Profile()
    for iteration in tqdm(range(starting_iteration, ITERATIONS), initial=starting_iteration, total=ITERATIONS):
        if iteration == 200:
            pr.enable()

        moves_memory = []

        epoch_scores = []

        states = torch.zeros((MAX_STEPS_PER_GAME, PARALLEL_GAMES_COUNT, STATE_SIZE)).to(device)
        actions = torch.zeros((MAX_STEPS_PER_GAME, PARALLEL_GAMES_COUNT)).to(device)
        logprobs = torch.zeros((MAX_STEPS_PER_GAME, PARALLEL_GAMES_COUNT)).to(device)
        rewards = torch.zeros((MAX_STEPS_PER_GAME, PARALLEL_GAMES_COUNT)).to(device)
        estimated_values = torch.zeros((MAX_STEPS_PER_GAME, PARALLEL_GAMES_COUNT)).to(device)

        policy_model.eval()
        value_model.eval()
        with torch.no_grad():
            games = [Game(tileset, device) for i in range(PARALLEL_GAMES_COUNT)]
            for step_number in range(MAX_STEPS_PER_GAME):
                step_states = torch.stack([game.get_state() for game in games]).to(device)
                states[step_number] = step_states

                step_logits = policy_model(step_states)
                estimated_values[step_number] = value_model(step_states).flatten()

                # probs = torch.distributions.categorical.Categorical(logits=step_logits)
                # valid_probs = torch.distributions.categorical.Categorical(probs=probs.probs * step_action_mask)
                # step_actions = valid_probs.sample()
                # logprobs[step_number] = valid_probs.log_prob(step_actions)

                probs = torch.distributions.categorical.Categorical(probs=step_logits)
                step_actions = probs.sample()
                actions[step_number] = step_actions
                logprobs[step_number] = probs.log_prob(step_actions)
                step_probs = probs

                for game_number, game in enumerate(games):
                    initial_score = game.get_score()

                    action = step_actions[game_number].item()
                    is_legal = game.apply_action(action)

                    new_score = game.get_score()
                    if is_legal:
                        rewards[step_number, game_number] = new_score - initial_score
                    else: 
                        rewards[step_number, game_number] = 0
                    
            final_states = torch.stack([game.get_state() for game in games]).to(device)
            # final_scores = torch.Tensor([game.get_score() for game in games]).to(device)
            final_is_done = torch.Tensor([game.is_done() for game in games]).to(device)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for step_number in reversed(range(MAX_STEPS_PER_GAME)):
                is_last_step = step_number == MAX_STEPS_PER_GAME - 1
                if is_last_step:
                    nextnonterminal = 1.0 - final_is_done
                    # next_return = value_model(final_states).flatten() # this estimates the value of the final state, in case it's not terminal
                    next_values = 0.0
                else:
                    nextnonterminal = 1.0
                    next_values = estimated_values[step_number + 1]
                
                delta = rewards[step_number] + DISCOUNT_RATE * next_values * nextnonterminal - estimated_values[step_number]
                lastgaelam = delta + DISCOUNT_RATE * 0.95 * nextnonterminal * lastgaelam
                advantages[step_number] = lastgaelam
            returns = advantages + estimated_values

            for game_number, game in enumerate(games):
                # score = final_scores[game_number].item()
                score = game.get_score()
                epoch_scores.append(score)

        s_optimize = time.time()

        epoch_loss = []

        if iteration == 200:
            print(time.time() - s_optimize)
            pr.disable()
            pr.dump_stats(f"profile_{time.time()}")

        # We don't need the step and game dimensions anymore
        b_states = states.reshape((-1,STATE_SIZE))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = estimated_values.reshape(-1)

        policy_model.train()
        value_model.train()

        b_inds = np.arange(BATCH_SIZE)
        for realepoch in range(EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb_inds = b_inds[start:start + MINIBATCH_SIZE]

                mb_states = b_states[mb_inds]
                mb_actions = b_actions[mb_inds]

                # logits = policy_model(mb_states)
                # probs = torch.distributions.categorical.Categorical(logits=logits)
                # valid_probs = torch.distributions.categorical.Categorical(probs=probs.probs * mb_action_masks)
                # newlogprobs = valid_probs.log_prob(mb_actions)
                # entropy = valid_probs.entropy()

                logits = policy_model(mb_states)
                # probs = torch.distributions.categorical.Categorical(probs=logits * mb_action_masks)
                probs = torch.distributions.categorical.Categorical(probs=logits)
                newlogprobs = probs.log_prob(mb_actions)
                entropy = probs.entropy()

                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_loss = pg_loss - ENTROPY_COEF * entropy.mean()

                policy_optimizer.zero_grad()
                pg_loss.backward()
                # pg_loss1.mean().backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 0.5)
                policy_optimizer.step()

                # Value loss
                newvalue = value_model(mb_states)
                newvalue = newvalue.view(-1)
                value_loss = value_loss_fn(newvalue, b_returns[mb_inds])

                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_model.parameters(), 0.5)
                value_optimizer.step()

        score_smoother.append(median(epoch_scores))
        median_score_smooth = float(sum(score_smoother)) / len(score_smoother)
        policy_scheduler.step()
        value_scheduler.step()

        print(f"Avg score: {float(sum(epoch_scores)) / len(epoch_scores)}")
        writer.add_scalar("Avg score", float(sum(epoch_scores)) / len(epoch_scores), iteration)
        print(f"Median score: {median(epoch_scores)}")
        writer.add_scalar("Median score", median(epoch_scores), iteration)
        print(f"Min score: {min(epoch_scores)}")
        writer.add_scalar("Min score", min(epoch_scores), iteration)
        print(f"Max score: {max(epoch_scores)}")
        writer.add_scalar("Max score", max(epoch_scores), iteration)
        print()
        print(f"Learning rate: {policy_scheduler.get_last_lr()[0]}")
        writer.add_scalar("Learning rate", policy_scheduler.get_last_lr()[0], iteration)
        # print(f"Avg loss: {float(sum(epoch_loss)) / len(epoch_loss)}")
        #writer.add_scalar("Avg loss", float(sum(epoch_loss)) / len(epoch_loss), epoch)
        #print(f"Median loss: {median(epoch_loss)}")
        #writer.add_scalar("Median loss", median(epoch_loss), epoch)
        print(f"Optimize: {time.time() - s_optimize}s")

        if iteration % 5000 == 0 and iteration > 0 and iteration != starting_iteration:
            prefix = f"takeiteasy_{start_timestamp}"
            specific_outdir = os.path.join(OUT_DIR, prefix)
            os.makedirs(specific_outdir, exist_ok=True)

            with gzip.GzipFile(os.path.join(specific_outdir, f"{prefix}_{iteration}.pth.gz"), 'wb') as f:
                torch.save({
                'iteration': iteration,
                'policy': {
                    'model_state_dict': policy_model.state_dict(),
                    'optimizer_state_dict': policy_optimizer.state_dict(),
                    'scheduler_state_dict': policy_scheduler.state_dict(),
                },
                'value': {
                    'model_state_dict': policy_model.state_dict(),
                    'optimizer_state_dict': policy_optimizer.state_dict(),
                    'scheduler_state_dict': policy_scheduler.state_dict(),
                },
                'start_timestamp': start_timestamp,
                }, f)

    writer.close()
