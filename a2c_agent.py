import torch
import torch.nn as nn
import torch.optim as optim

class ActorCriticAgent:
    def __init__(self, state_size, action_size, gamma, learning_rate_actor, learning_rate_critic, entropy_beta, entropy_beta_min, entropy_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.entropy_beta_min = entropy_beta_min
        self.entropy_decay = entropy_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

    def act(self, state, invalid_actions=None):
        state_tensor = torch.Tensor(state).unsqueeze(0).to(self.device)

        # Get probabilities of each action
        action_probs = self.actor(state_tensor)

        # Mask invalid actions
        if invalid_actions:
            mask = torch.ones_like(action_probs, device=self.device)
            for action in invalid_actions:
                mask[0][action] = 0

            masked_probs = action_probs * mask
            
            # Re-normalize probabilities (sum = 1)
            normalized_probs = masked_probs / masked_probs.sum()
            # Get tensor of probabilities => distribution object
            action_dist = torch.distributions.Categorical(normalized_probs)
        else:
            action_dist = torch.distributions.Categorical(action_probs)

        entropy = action_dist.entropy()
        
        # Get random action according to the distribution
        action = action_dist.sample()

        # Return action and log of its probability and entropy
        return action.item(), action_dist.log_prob(action), entropy
    
    def compute_loss(self, log_probs, values, rewards, dones, entropies):
        # Discounted cumulative rewards
        returns = []
        R = 0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)

        # Normalize returns
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Values predicted by the critic during the episode
        values = torch.cat(values).squeeze().to(self.device)
        # Actual return - predicted values
        # Advantage is how much better / worse the taken action was compared to
        # the critic's estimate
        advantages = returns - values

        # Policy gradient theorem, -log(prob of taking action at given state st) * advatange of action at given state st
        # Increase probability of actions with positive advantage
        entropies = torch.tensor(entropies).to(self.device)
        actor_loss = -torch.sum(torch.cat(log_probs) * advantages.detach()) - self.entropy_beta * torch.sum(entropies)

        # Minimize error between predicted values and actual returns
        # MSE between actual return and predicted value then sum
        # Critic learns to predict expected value more accurately
        critic_loss = torch.sum(advantages.pow(2))

        return actor_loss, critic_loss
    
    def decay_entropy_beta(self):
        if self.entropy_beta > self.entropy_beta_min:
            self.entropy_beta *= self.entropy_decay

    def act_greedy(self, state, invalid_actions=None):
        state_tensor = torch.Tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        if invalid_actions:
            mask = torch.ones_like(action_probs, device=self.device)
            for action in invalid_actions:
                mask[0][action] = 0

            masked_probs = action_probs * mask
            action = torch.argmax(masked_probs).item()
        else:
            action = torch.argmax(action_probs).item()

        return action


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        input_channels = state_size[0]
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=2, stride=1)
        
        self.fc1 = nn.Linear(384, 432)
        self.fc2 = nn.Linear(432, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        input_channels = state_size[0]

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=144, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=288, out_channels=432, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(432, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        state_value = self.fc2(x)
        return state_value
    

def train_a2c(agent, env, num_episodes, save_frequency=100):
    best_episode, tile, highest_score = 0, 0, 0

    best_models = []
    max_tile_512 = 0
    max_tile_1024 = 0
    total_rewards, total_actor_losses, total_critic_losses, highest_tiles, total_scores, total_entropy_betas = [], [], [], [], [], []
    agent.actor.train()
    agent.critic.train()

    for episode in range(num_episodes):
        env.reset()

        state = env.get_log_board()
        rewards, log_probs, values, dones, entropies = [], [], [], [], []
        total_reward = 0

        while not env.is_game_over():
            invalid_actions = env.get_invalid_actions()
            action, log_prob, entropy = agent.act(state, invalid_actions)

            next_state, reward, done, info = env.step(action)

            state_tensor = torch.Tensor(state).unsqueeze(0).to(agent.device)
            value = agent.critic(state_tensor)

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            entropies.append(entropy)

            total_reward += reward
            state = next_state

        actor_loss, critic_loss = agent.compute_loss(log_probs, values, rewards, dones, entropies)
        
        score = info.get('total_score', 0)
        print(f"Episode: {episode}, Total Reward: {total_reward:.4f}, Highest Tile: {info.get('highest_tile', 0)}, Score: {score:.4f}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy Beta: {agent.entropy_beta:.4f}")
        
        total_rewards.append(total_reward)
        total_actor_losses.append(actor_loss.item())
        total_critic_losses.append(critic_loss.item())
        highest_tiles.append(info.get('highest_tile', 0))
        total_scores.append(score)
        total_entropy_betas.append(agent.entropy_beta)

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        agent.decay_entropy_beta()

        if score > highest_score:
            best_episode, tile, highest_score = episode, info.get('highest_tile', 0), score

        if info.get('highest_tile', 0) == 512:
            max_tile_512 += 1

        if info.get('highest_tile', 0) == 1024:
            max_tile_1024 += 1

        if len(best_models) < 3 or score > best_models[0][0]:
            best_models.append((score, episode, agent.actor.state_dict()))

            best_models = sorted(best_models, key=lambda x: x[0])
            if len(best_models) > 3:
                best_models.pop(0)
                
        if episode % save_frequency == 0:
            path = f'/kaggle/working/{episode}.pt'
            torch.save(agent.actor.state_dict(), path)

    for rank, (score, episode, state_dict) in enumerate(reversed(best_models), start=1):
        path = f'/kaggle/working/{rank}.pt'
        torch.save(state_dict, path)
        print(f'Saved model, episode: {episode}, score: {score}')

    return best_episode, tile, highest_score, best_models, max_tile_512, max_tile_1024, total_rewards, total_actor_losses, total_critic_losses, highest_tiles, total_scores, total_entropy_betas


def test_a2c(agent, env, num_episode):
    best_episode, tile, highest_score = 0, 0, 0

    max_tile_512 = 0
    max_tile_1024 = 0
    agent.actor.eval()

    for episode in range(num_episode):
        env.reset()

        state = env.get_log_board()
        total_reward = 0

        while not env.is_game_over():
            invalid_actions = env.get_invalid_actions()

            with torch.no_grad():
                action, _, _ = agent.act(state, invalid_actions)

            _, reward, _, info = env.step(action)
            total_reward += reward

        score = info.get('total_score', 0)
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Highest Tile: {info.get('highest_tile', 0)}, Score: {score}")

        if score > highest_score:
            best_episode, tile, highest_score = episode, info.get('highest_tile', 0), score
        
        if info.get("highest_tile", 0) == 512:
            max_tile_512 += 1
        if info.get("highest_tile", 0) == 1024:
            max_tile_1024 += 1

    return best_episode, tile, highest_score, max_tile_512, max_tile_1024
    