#!/usr/bin/env python3
"""
Training with learned card embeddings and attention mechanism.
Let the network learn what features matter.
"""

import pickle
import numpy as np
from collections import deque
from typing import List, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from scoundrel_env import ScoundrelEnv, CardType


class CardEmbeddingAttentionModel(nn.Module):
    """
    Model that learns card embeddings and uses attention.

    Input: Raw card data instead of hand-crafted features
    - Room cards (4 cards, each with type + value)
    - Weapon (type + value + max_monster)
    - Health
    - Deck size
    - Can skip flag
    """

    def __init__(self, embed_dim=32, num_heads=4, hidden_dim=256):
        super().__init__()

        # Card type embedding (0=empty, 1=monster, 2=weapon, 3=potion)
        self.type_embed = nn.Embedding(4, embed_dim)

        # Card value embedding (0-15, where 0=no card)
        self.value_embed = nn.Embedding(16, embed_dim)

        # Combine type + value embeddings
        self.card_proj = nn.Linear(embed_dim * 2, embed_dim)

        # Self-attention over cards
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Global game state features
        self.state_proj = nn.Linear(5, embed_dim)  # health, weapon_val, weapon_max, deck_size, can_skip

        # Combine attended cards + state
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim // 2, 5)
        self.critic = nn.Linear(hidden_dim // 2, 1)

    def encode_cards(self, room_cards, weapon_info):
        """
        Encode cards using learned embeddings.

        room_cards: (batch, 4, 2) - 4 cards, each with [type, value]
        weapon_info: (batch, 3) - [type, value, max_monster]
        """
        batch_size = room_cards.shape[0]

        # Embed room cards
        room_types = room_cards[:, :, 0].long()  # (batch, 4)
        room_values = room_cards[:, :, 1].long()  # (batch, 4)

        type_emb = self.type_embed(room_types)  # (batch, 4, embed_dim)
        value_emb = self.value_embed(room_values)  # (batch, 4, embed_dim)

        card_emb = torch.cat([type_emb, value_emb], dim=-1)  # (batch, 4, embed_dim*2)
        card_emb = self.card_proj(card_emb)  # (batch, 4, embed_dim)

        # Embed weapon as 5th "card"
        weapon_type = weapon_info[:, 0:1].long()  # (batch, 1)
        weapon_value = weapon_info[:, 1:2].long()  # (batch, 1)

        weapon_type_emb = self.type_embed(weapon_type)  # (batch, 1, embed_dim)
        weapon_value_emb = self.value_embed(weapon_value)  # (batch, 1, embed_dim)

        weapon_emb = torch.cat([weapon_type_emb, weapon_value_emb], dim=-1)
        weapon_emb = self.card_proj(weapon_emb)  # (batch, 1, embed_dim)

        # Combine all cards for attention
        all_cards = torch.cat([card_emb, weapon_emb], dim=1)  # (batch, 5, embed_dim)

        return all_cards

    def forward(self, room_cards, weapon_info, game_state):
        """
        Forward pass.

        room_cards: (batch, 4, 2)
        weapon_info: (batch, 3)
        game_state: (batch, 5) - [health, weapon_val, weapon_max, deck_pct, can_skip]
        """
        # Encode cards
        card_embeddings = self.encode_cards(room_cards, weapon_info)

        # Self-attention over cards
        attended, _ = self.attention(card_embeddings, card_embeddings, card_embeddings)

        # Pool attended cards (mean)
        pooled = attended.mean(dim=1)  # (batch, embed_dim)

        # Encode game state
        state_emb = self.state_proj(game_state)  # (batch, embed_dim)

        # Combine
        combined = torch.cat([pooled, state_emb], dim=-1)  # (batch, embed_dim*2)
        features = self.combiner(combined)

        return self.actor(features), self.critic(features)

    def get_action(self, obs_dict, valid_actions: List[int]):
        """Get action from observation dictionary."""
        room_cards = torch.tensor(obs_dict['room_cards']).unsqueeze(0).float()
        weapon_info = torch.tensor(obs_dict['weapon_info']).unsqueeze(0).float()
        game_state = torch.tensor(obs_dict['game_state']).unsqueeze(0).float()

        logits, value = self.forward(room_cards, weapon_info, game_state)
        logits = logits.squeeze(0)
        value = value.squeeze()

        mask = torch.full((5,), float('-inf'))
        for a in valid_actions:
            mask[a] = 0
        masked_logits = logits + mask

        probs = torch.softmax(masked_logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value, dist.entropy()


class AttentionEnv:
    """Environment wrapper that provides raw card data."""

    def __init__(self, seed=None):
        self.env = ScoundrelEnv(seed=seed)
        self.rooms_cleared = 0

    def reset(self, seed=None):
        self.env.reset(seed)
        self.rooms_cleared = 0
        return self._get_obs()

    @property
    def done(self):
        return self.env.done

    @property
    def won(self):
        return self.env.won

    @property
    def health(self):
        return self.env.health

    def get_valid_actions(self):
        return self.env.get_valid_actions()

    def calculate_score(self):
        if self.env.won:
            return float(self.env.health)
        else:
            remaining = sum(c.value for c in self.env.deck if c.card_type == CardType.MONSTER)
            remaining += sum(c.value for c in self.env.room if c.card_type == CardType.MONSTER)
            return float(self.env.health - remaining)

    def _get_obs(self):
        """Get observation as raw card data."""
        # Room cards: [type, value] for each of 4 cards
        room_cards = []
        sorted_room = sorted(self.env.room, key=lambda c: (c.card_type.value, c.value))
        for i in range(4):
            if i < len(sorted_room):
                card = sorted_room[i]
                card_type = card.card_type.value + 1  # 1=monster, 2=weapon, 3=potion
                card_value = card.value
            else:
                card_type = 0  # empty
                card_value = 0
            room_cards.append([card_type, card_value])

        # Weapon info: [type, value, max_monster]
        if self.env.weapon:
            weapon_type = 2  # weapon type
            weapon_value = self.env.weapon.value
            weapon_max = self.env.weapon.max_monster if self.env.weapon.max_monster else 15
        else:
            weapon_type = 0
            weapon_value = 0
            weapon_max = 0
        weapon_info = [weapon_type, weapon_value, weapon_max]

        # Game state: [health, weapon_val_norm, weapon_max_norm, deck_pct, can_skip]
        game_state = [
            self.env.health / 20.0,
            weapon_value / 10.0,
            weapon_max / 15.0,
            len(self.env.deck) / 44.0,
            0.0 if self.env.last_room_skipped else 1.0
        ]

        return {
            'room_cards': np.array(room_cards, dtype=np.float32),
            'weapon_info': np.array(weapon_info, dtype=np.float32),
            'game_state': np.array(game_state, dtype=np.float32)
        }

    def step(self, action):
        old_health = self.env.health
        old_deck_size = len(self.env.deck) + len(self.env.room)

        self.env.step(action)
        obs = self._get_obs()

        reward = 0.0

        # Progress reward
        cards_cleared = old_deck_size - (len(self.env.deck) + len(self.env.room))
        if cards_cleared > 0:
            progress_fraction = 1 - len(self.env.deck) / 44.0
            reward += 0.05 * cards_cleared * (1 + progress_fraction)
            self.rooms_cleared += 1

        # Health management
        health_change = self.env.health - old_health
        if health_change < 0:
            health_fraction = old_health / 20.0
            penalty = 0.03 * abs(health_change) * (2 - health_fraction)
            reward -= penalty
        elif health_change > 0:
            reward += 0.02 * health_change

        # Terminal rewards
        if self.env.done:
            if self.env.won:
                reward += 100.0 + self.env.health * 5.0
            else:
                score = self.calculate_score()
                progress = self.rooms_cleared / 12.0
                reward += score / 50.0
                reward += progress * 0.5

        return obs, reward, self.env.done, {}


def collect_episode(env, model):
    """Collect a single episode."""
    obs_list = []
    log_probs = []
    values = []
    rewards = []
    entropies = []

    obs = env.reset()

    while not env.done:
        valid_actions = env.get_valid_actions()
        action, log_prob, value, entropy = model.get_action(obs, valid_actions)

        obs_list.append(obs)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

    return obs_list, log_probs, values, rewards, entropies, env.calculate_score(), env.won


def train_batch(model, optimizer, batch_data, gamma=0.99, entropy_coef=0.02):
    """Train on a batch of episodes."""
    all_returns = []
    all_advantages = []
    all_log_probs = []
    all_values = []
    all_entropies = []

    for log_probs, values, rewards, entropies in batch_data:
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        values_t = torch.stack(values)
        advantages = returns - values_t.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_returns.append(returns)
        all_advantages.append(advantages)
        all_log_probs.append(torch.stack(log_probs))
        all_values.append(values_t)
        all_entropies.append(torch.stack(entropies))

    returns = torch.cat(all_returns)
    advantages = torch.cat(all_advantages)
    log_probs = torch.cat(all_log_probs)
    values = torch.cat(all_values)
    entropies = torch.cat(all_entropies)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = 0.5 * ((returns - values) ** 2).mean()
    entropy_loss = -entropy_coef * entropies.mean()

    total_loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return total_loss.item()


def evaluate(model, num_episodes=1000):
    """Evaluate the model."""
    wins = 0
    total_score = 0

    model.eval()
    for seed in range(num_episodes):
        env = AttentionEnv(seed=seed)
        obs = env.reset()

        while not env.done:
            valid_actions = env.get_valid_actions()

            room_cards = torch.tensor(obs['room_cards']).unsqueeze(0).float()
            weapon_info = torch.tensor(obs['weapon_info']).unsqueeze(0).float()
            game_state = torch.tensor(obs['game_state']).unsqueeze(0).float()

            with torch.no_grad():
                logits, _ = model(room_cards, weapon_info, game_state)
                logits = logits.squeeze(0)

            mask = torch.full((5,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0
            action = (logits + mask).argmax().item()

            obs, _, _, _ = env.step(action)

        if env.won:
            wins += 1
            total_score += env.health
        else:
            total_score += env.calculate_score()

    model.train()
    return wins / num_episodes, total_score / num_episodes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Attention Model Training')
    parser.add_argument('--trajectories', type=str, default='winning_trajectories_v7.pkl')
    parser.add_argument('--episodes', type=int, default=2000000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--eval-interval', type=int, default=25000)
    parser.add_argument('--winning-seed-ratio', type=float, default=0.20)
    parser.add_argument('--output', type=str, default='rl_attention.pt')
    args = parser.parse_args()

    print("Loading winning trajectories...")
    with open(args.trajectories, 'rb') as f:
        trajectories = pickle.load(f)
    winning_seeds = [t['seed'] for t in trajectories]
    print(f"Loaded {len(winning_seeds)} winning seeds")

    print(f"\n=== ATTENTION MODEL TRAINING ===")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Attention heads: {args.num_heads}")
    print(f"Episodes: {args.episodes:,}")
    print()

    model = CardEmbeddingAttentionModel(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)
    best_win_rate = 0
    best_avg_score = float('-inf')
    best_model_state = None

    wins_on_winning_seeds = 0
    attempts_on_winning_seeds = 0

    start_time = time.time()
    episode = 0

    entropy_start = 0.05
    entropy_end = 0.01

    while episode < args.episodes:
        progress = episode / args.episodes
        entropy_coef = entropy_start + (entropy_end - entropy_start) * progress

        batch_data = []

        for _ in range(args.batch_size):
            use_winning_seed = np.random.random() < args.winning_seed_ratio
            if use_winning_seed and winning_seeds:
                seed = int(np.random.choice(winning_seeds))
                attempts_on_winning_seeds += 1
            else:
                seed = None

            env = AttentionEnv(seed=seed)
            obs_list, log_probs, values, rewards, entropies, score, won = \
                collect_episode(env, model)

            batch_data.append((log_probs, values, rewards, entropies))

            if use_winning_seed and won:
                wins_on_winning_seeds += 1

            recent_scores.append(score)
            recent_wins.append(1 if won else 0)

            episode += 1

        train_batch(model, optimizer, batch_data, entropy_coef=entropy_coef)

        if episode % args.eval_interval < args.batch_size:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            remaining = (args.episodes - episode) / eps_per_sec

            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)
            eval_win_rate, eval_avg_score = evaluate(model, num_episodes=2000)

            print(f"Episode {episode:,} ({100*episode/args.episodes:.1f}%):")
            print(f"  Training: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")
            print(f"  Entropy:  {entropy_coef:.4f}")
            if attempts_on_winning_seeds > 0:
                seed_win_rate = wins_on_winning_seeds / attempts_on_winning_seeds
                print(f"  Winning seeds: {wins_on_winning_seeds}/{attempts_on_winning_seeds} ({seed_win_rate:.2%})")
            print(f"  Speed: {eps_per_sec:.1f} eps/sec, ETA: {remaining/3600:.1f}h")

            if eval_win_rate > best_win_rate or (eval_win_rate == best_win_rate and eval_avg_score > best_avg_score):
                best_win_rate = eval_win_rate
                best_avg_score = eval_avg_score
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  *** New best! win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f} ***")

            print()

    if best_model_state:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Total time: {total_time/3600:.2f} hours")

    print("\n=== Final Evaluation (10k episodes) ===")
    final_win_rate, final_avg_score = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg score: {final_avg_score:.1f}")

    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'avg_score': final_avg_score,
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
