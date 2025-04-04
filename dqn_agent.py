import numpy as np
import random
from collections import deque
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

class DQNAgent:
    def __init__(self, state_size, action_size, gamma, epsilon, epsilon_min, 
                 epsilon_decay, learning_rate, batch_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.999)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, invalid_actions=None):
        # Pick random action (explore) out of the valid ones
        if np.random.rand() <= self.epsilon:
            valid_actions = [a for a in range(self.action_size) if a not in invalid_actions]
            return np.random.choice(valid_actions)

        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state).squeeze(0)

        # Mask invalid actions
        if invalid_actions is not None:
            q_values[invalid_actions] = float('-inf')

        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.Tensor(np.array(states)).to(self.device)
        next_states = torch.Tensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Predict Q values for each action with the target model (batch_size, action_size)
        new_q_values = self.target_model(next_states)
        # Get max Q value from all actions for each state
        max_new_q_values = torch.max(new_q_values, dim=1)[0]
        # Compute Q values with Bellman equation 
        target_q_values = rewards + self.gamma * max_new_q_values * (1 - dones)

        # Get the Q values for each action with model (batch_size, action_size)
        q_values = self.model(states)
        # Get Q values for the taken actions
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))

    def act_greedy(self, state, invalid_actions=None):
        state_tensor = torch.Tensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).detach().numpy().squeeze(0)

        # Mask invalid actions
        q_values_copy = np.copy(q_values)
        q_values_copy[invalid_actions] = float('-inf')

        action = np.argmax(q_values_copy)
        return action
    

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        input_channels = state_size[0]
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=144, kernel_size=2, stride=1)  # Updated to match checkpoint
        self.conv2 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=2, stride=1)  # Updated to match checkpoint
        self.conv3 = nn.Conv2d(in_channels=288, out_channels=432, kernel_size=2, stride=1)  # Updated to match checkpoint
        
        self.fc1 = nn.Linear(432, 512)  # Updated to match checkpoint
        self.fc2 = nn.Linear(512, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    

def train_dqn(agent, env, num_episodes, replay_frequency=10, target_update_frequency=300):
    best_episode, tile, highest_score = 0, 0, 0
    best_models = []
    agent.model.train()

    target_step = 0
    
    total_rewards, total_losses, highest_tiles, total_scores, epsilons, learning_rates = [], [], [], [], [], []
    
    for episode in range(num_episodes):
        env.reset()

        total_reward = 0
        step = 0

        while not env.is_game_over():
            state = env.get_log_board()
            invalid_actions = env.get_invalid_actions()

            action = agent.act(state, invalid_actions)

            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward

            if step % replay_frequency == 0:
                loss = agent.replay()
                if loss is not None:
                    total_losses.append(loss)
                    total_rewards.append(total_reward)
                    total_scores.append(info.get('total_score', 0))
                    highest_tiles.append(info.get('highest_tile', 0))
                    epsilons.append(agent.epsilon)
                    learning_rates.append(agent.scheduler.get_last_lr()[0])

            step +=1
            target_step += 1
            
        # if target_step >= target_update_frequency:
        #     agent.update_target_model()
        #     target_step = 0

        if episode % 10 == 0:
            agent.update_target_model()

        agent.decay_epsilon()

        print(f'Episode: {episode}, Total Reward: {total_reward:.4f}, Highest Tile: {info.get("highest_tile", 0)}, Score: {info.get("total_score", 0):.4f}, Epsilon: {agent.epsilon:.6f}, Learning Rate: {agent.scheduler.get_last_lr()[0]:.6f}')
        
        score = info.get('total_score', 0)
        if score > highest_score:
            best_episode, tile, highest_score = episode, info.get('highest_tile', 0), score

        if len(best_models) < 8 or score > best_models[0][0]:
            best_models.append((score, episode, agent.model.state_dict()))

            best_models = sorted(best_models, key=lambda x: x[0])
            if len(best_models) > 8:
                best_models.pop(0)

    for rank, (score, episode, state_dict) in enumerate(reversed(best_models), start=1):
        path = f'2048/models/{rank}.pt'
        torch.save(state_dict, path)
        print(f'Saved model, episode: {episode}, score: {score}')

    return (best_episode, tile, highest_score, best_models, total_rewards, total_losses, highest_tiles, total_scores, epsilons, learning_rates)


def test_dqn(agent, env, num_episodes):
    best_episode, tile, highest_score = 0, 0, 0
    agent.model.eval()
    max_tile_2048 = 0
    max_tile_1024 = 0
    max_tile_512 = 0

    for episode in range(num_episodes):
        env.reset()

        total_reward = 0

        while not env.is_game_over():
            state = env.get_log_board()
            invalid_actions = env.get_invalid_actions()
            state_tensor = torch.Tensor(state).unsqueeze(0)

            with torch.no_grad():
                q_values = agent.model(state_tensor).detach().numpy().squeeze(0)

            # Mask invalid actions
            q_values_copy = np.copy(q_values)
            q_values_copy[invalid_actions] = float('-inf')

            action = np.argmax(q_values_copy)
            _, reward, _, info = env.step(action)

            total_reward += reward

        print('Episode: {},  Total Reward: {:.4f},  Highest Tile: {},  Score: {}'.format(
            episode, total_reward, info.get('highest_tile', 0), info.get('total_score', 0)))
        
        if info.get('total_score', 0) > highest_score:
                best_episode, tile, highest_score = episode, info.get('highest_tile', 0), info.get('total_score', 0)
                
        if info.get('highest_tile', 0) == 2048:
            max_tile_2048 += 1
        elif info.get('highest_tile', 0) == 1024:
            max_tile_1024 += 1
        elif info.get('highest_tile', 0) == 512:
            max_tile_512 += 1

    return (best_episode, tile, highest_score, max_tile_2048, max_tile_1024, max_tile_512)