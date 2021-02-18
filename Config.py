# -*- coding: utf-8 -*-

SEED = None  # None if we want a random seed or a number if we want to specify a seed

LOAD_MAP_FILE = False  # For loading map from a file

# FILES PATH
BASE_ROUTE = './'
DATA_ROUTE = 'data.json'
MAP_ROUTE = 'map.txt'
EPSILONS_ROUTE = 'epsilons.csv'
SIMULATIONS_TIMES_ROUTE = 'simulations_times.csv'
SIMULATIONS_REWARDS = 'simulations_rewards.csv'
SIMULATIONS_COVERAGES = 'simulations_coverages.csv'
SIMULATION_COVERAGES = 'simulation_coverages.csv'

# Episode configuration
SIMULATIONS = 500
PRINT_SIMULATIONS = False  # Print for checkpoint on console
SIMULATIONS_CHECKPOINT = 100  # Number of episodes that must happen before printing checkpoint

# Environment configuration
ENVIRONMENT_ROWS = 20
ENVIRONMENT_COLUMNS = 20
SQUARE = True  # Make environment polygon squared
START_CORNER_0_0 = True

# Agent Configuration
ACTIONS_DICT = {
    0: 'left',
    1: 'up',
    2: 'right',
    3: 'down',
}
NEW_CELL_REWARD = 225.7425433
VISITED_CELL_REWARD = -138.2603908
NO_CELL_REWARD = -333.3099984

# Learning process configuration
GLOBAL_MODEL = False
EPSILON = 0.487333128124803
MIN_EPSILON = 0.05
GAMMA = 0.996957222
MEMORY_SIZE = 10
BATCH_SIZE = 44
EPOCHS = 10
VERBOSE = 0
EPSILON_DECAY = 0.906585414

# Model
DENSE1_SIZE = 662
DENSE1_ACTIVATION = 'sigmoid'
DENSE2_SIZE = len(ACTIONS_DICT)
DENSE2_ACTIVATION = 'sigmoid'
OPTIMIZER = 'Adadelta'

# Early experiment stopping
MAXIMUM_WAIT_HOURS = 0.5
COMPLETENESS_COVERAGE = 1.0  # float between 0 and 1
MAXIMUM_UNCHANGED_ENVIRONMENT_EPISODES = 9999999999

# Unit tests
UNIT_TESTS = False