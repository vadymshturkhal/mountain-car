# Files
CAR_WEIGHTS_FILENAME = './model/model_car.pth'
FOOD_WEIGHTS_FILENAME = './model/model_food.pth'
SCORE_DATA_FILENAME = './data/latest.csv'

# Screen
SCREEN_W = 210
SCREEN_H = 210

# Game
BLOCK_SIZE = 30
DIRECTIONS_QUANTITY = 4
FRAME_RESTRICTION = 300

# Speed
GAME_SPEED = 40
CAR_SPEED = 0.000004
FOOD_SPEED_MULTIPLIER = 2

# Train
CAR_ACTION_LENGTH = 3
DROPOUT_RATE = 0.2
LR = 0.0001
WEIGHT_DECAY = 1e-5
CAR_GAMMA = 0.9
CAR_START_EPSILON = 1
CAR_MIN_EPSILON = 0.1
EPSILON_SHIFT = 0
CAR_ENERGY = 400
TRAINER_STEPS = 4

# CAR layers
CAR_INPUT_LAYER_SIZE = 2
CAR_HIDDEN_LAYER_SIZE1 = 256
CAR_HIDDEN_LAYER_SIZE2 = 256
CAR_OUTPUT_LAYER_SIZE = 3

# CAR rewards
REWARD_WIN = 10
REWARD_MOVE = -1
