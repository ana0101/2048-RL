import numpy as np

class Game:
    def __init__(self, size=4):
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.possible_values = [0] + [2**i for i in range(1, int(np.log2(2048)) + 1)]
        self.add_tile()
        self.add_tile()

    def show_board(self):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                print(f'{self.board[i][j]:^3}', end=' ')
            print()
        print()

    def get_one_hot_board(self):
        one_hot = np.zeros((len(self.board), len(self.board), len(self.possible_values)), dtype=int)

        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i, j] != 0:
                    one_hot[i, j, int(np.log2(self.board[i, j]))] = 1
        
        one_hot_permuted = np.transpose(one_hot, (2, 0, 1))
        return one_hot_permuted
    
    def get_log_board(self):
        log_board = np.zeros_like(self.board)
        non_zero_board = self.board > 0
        log_board[non_zero_board] = np.log2(self.board[non_zero_board])
        # Add another dimension for channel (for convolutional network) => (1, 4, 4)
        return np.expand_dims(log_board, axis=0)
    
    def add_tile(self):
        # Get the empty tiles
        empty_tiles = [(i, j) for i in range(len(self.board)) for j in range(len(self.board)) if self.board[i][j] == 0]
        if empty_tiles:
            # Pick a random empty tile
            i, j = empty_tiles[np.random.choice(len(empty_tiles))]
            # 2 with 90% probability, 4 with 10% probability
            if np.random.random() < 0.9:
                self.board[i][j] = 2
            else:
                self.board[i][j] = 4

    def move(self, direction):
        score = 0
        # moved_cells = 0
        board_copy = np.copy(self.board)
        # Rotate the board
        if direction == 0:     # up
            self.board = np.rot90(self.board, 1)
        elif direction == 1:   # down
            self.board = np.rot90(self.board, 3)
        elif direction == 2:   # right
            self.board = np.rot90(self.board, 2)

        # Move the tiles
        for i in range(len(self.board)):
            self.board[i], s = self.move_tiles_left(self.board[i])
            score += s
            # moved_cells += moved

        # Rotate back the board
        if direction == 0:
            self.board = np.rot90(self.board, 3)
        elif direction == 1:
            self.board = np.rot90(self.board, 1)
        elif direction == 2:
            self.board = np.rot90(self.board, 2)

        self.score += score

        # Check if invalid move 
        if np.array_equal(board_copy, self.board):
            return -1

        # Only add new tile if move is valid 
        self.add_tile()
        # Return points obtained from this round
        return score

    def move_tiles_left(self, row):
        # Move all tiles in the row to the left
        score = 0
        # moved_cells = 0
        # original_row = row.copy()

        # Get all the tiles in the row that are not zeros
        row = [tile for tile in row if tile != 0]
        # Merge adjacent tiles with the same value
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                # print('score in move tiles left: ', score)
                row[i + 1] = 0
        # Remove the zeros created by merging (move the remaining tiles to the left)
        row = [tile for tile in row if tile != 0]
        # Add zeros to the row
        row += [0] * (len(self.board) - len(row))

        # Count moved cells by comparing the new row to the original
        # for i in range(len(row)):
        #     if row[i] != original_row[i]:
        #         moved_cells += 1

        return row, score
    
    def get_invalid_actions(self):
        invalid_actions = []

        for action in range(4):
            board_copy = np.copy(self.board)
            score_copy = self.score

            if self.move(action) == -1:
                invalid_actions.append(action)
            
            self.board = board_copy
            self.score = score_copy
        
        return invalid_actions
    
    def is_game_over(self):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                # If there is an empty tile, the game is not over
                if self.board[i][j] == 0:
                    return False
                # If there are two adjacent tiles with the same value, the game is not over
                if j < len(self.board) - 1 and self.board[i][j] == self.board[i][j + 1]:
                    return False
                if i < len(self.board) - 1 and self.board[i][j] == self.board[i + 1][j]:
                    return False
        # Otherwise, the game is over
        return True
    
    def get_info(self):
        return {
            'empty_tiles': np.sum(self.board == 0),
            'highest_tile': np.max(self.board),
            'total_score': self.score,
        }
    
    def count_empty_tiles(self):
        return np.sum(self.board == 0)
    
    def reset(self):
        self.board = np.zeros((len(self.board), len(self.board)), dtype=int)
        self.score = 0
        self.add_tile()
        self.add_tile()

    def step(self, action):
        empty_tiles_before = self.count_empty_tiles()
        max_tile_before = np.max(self.board)

        # Return next state, reward, terminal, info
        reward = self.move(action)

        if reward >= 256:
            reward *= 2
        empty_tiles_after = self.count_empty_tiles()

        if max_tile_before < np.max(self.board):
            reward *= 2

        if reward < 0:
            reward = np.log2(reward)

        reward += 0.05 * (empty_tiles_after - empty_tiles_before)
        
        return self.get_log_board(), reward, self.is_game_over(), self.get_info()

    def play(self):
        self.reset()
        self.add_tile()
        self.add_tile()
        while not self.is_game_over():
            ok = False
            while not ok:
                print('Score:', self.score)
                self.show_board()
                action = int(input('Enter the action (0 = up, 1 = down, 2 = right, 3 = left): '))
                board, result, _, info = self.step(action)
                print(board.shape)
                if result != -1:
                    ok = True
                else:
                    print('Invalid action, please try again.')
        self.show_board()
