import pygame
from game import Game

SIZE = 4
TILE_SIZE = 100
TILE_MARGIN = 10
WINDOW_SIZE = SIZE * TILE_SIZE + (SIZE + 1) * TILE_MARGIN
FONT_SIZE = 40

BACKGROUND_COLOR = (187, 173, 160)
EMPTY_TILE_COLOR = (205, 193, 180)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}
TEXT_COLOR = (119, 110, 101)


def draw_board(game, screen, font):
    screen.fill(BACKGROUND_COLOR)
    for i in range(SIZE):
        for j in range(SIZE):
            value = game.board[i][j]
            rect = pygame.Rect(j * TILE_SIZE + (j + 1) * TILE_MARGIN,
                               i * TILE_SIZE + (i + 1) * TILE_MARGIN,
                               TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, TILE_COLORS.get(value, EMPTY_TILE_COLOR), rect)
            if value != 0:
                text = font.render(str(value), True, TEXT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
    pygame.display.flip()


def play_game_agent(agent):
    pygame.init()
    
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("2048")

    font = pygame.font.Font(None, FONT_SIZE)

    game = Game(SIZE)
    clock = pygame.time.Clock()

    running = True
    game_over = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            state = game.get_log_board()
            invalid_actions = game.get_invalid_actions()
            action = agent.act_greedy(state, invalid_actions)
            game.move(action)
            draw_board(game, screen, font)

            if game.is_game_over():
                game_over = True
                text = font.render("Game Over", True, (255, 0, 0))
                text_rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
                screen.blit(text, text_rect)
                pygame.display.flip()

        clock.tick(10)

    pygame.quit()
