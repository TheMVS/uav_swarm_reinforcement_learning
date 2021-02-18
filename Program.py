# -*- coding: utf-8 -*-

import Config
import cProfile
import re

cProfile.run('re.compile("foo|bar")')


class Program:
    'Common base class for all programs'
    __drones = []
    __agents = []
    __points = []
    __original_environment = None
    __operator_position = None
    __drone_initial_position = (0, 0)

    def __init__(self):
        return

    # Getters and setters
    def get_drones(self):
        return self.__drones

    def get_points(self):
        return self.__points

    def get_agents(self):
        return self.__agents

    def get_environment(self):
        return self.__original_environment

    def set_environment(self, environment):
        self.__original_environment = environment

    # Other methods
    def normalize_coordinate_value(self, value):
        # Shapely has float point precision errors
        min = -90.0
        max = 90.0

        num = float(value) - min
        denom = max - min

        return num / denom

    def denormalize_coordinate_value(self, norm_value):
        # Shapely has float point precision errors
        min = -90.0
        max = 90.0

        denom = max - min

        return float(norm_value) * denom + min

    def read_data(self):
        # Load data from JSON
        import json
        from Drone import Drone

        with open(Config.BASE_ROUTE + Config.DATA_ROUTE) as json_file:
            data = json.load(json_file)
            self.__drones = []
            for d in data['drones']:  # Drones info
                self.__drones.append(
                    Drone(d['name'].replace(" ", "_"), d['battery_time'], d['speed'],
                          (d['image_size']['w'], d['image_size']['h']),
                          d['height'], d['image_angle']))

            self.__points = []
            for p in data['points']:  # Map info
                self.__points.append(
                    (self.normalize_coordinate_value(p['lat']), self.normalize_coordinate_value(p['long'])))
            self.__agents = []

            from shapely.geometry import Point  # Operator's info
            self.__operator_position = Point((self.normalize_coordinate_value(data['operator_position']['lat']),
                                              self.normalize_coordinate_value(data['operator_position']['long'])))

    def compute_minimum_area(self, drones):
        # Get drones minimum image area (supposing a triangle composed of two rectangle triangles)
        areas = []
        for drone in drones:
            import numpy as np
            a = drone.get_height()
            A = np.deg2rad(drone.get_image_angle() / 2.0)
            C = np.deg2rad(90.0)
            B = np.deg2rad(180.0 - 90 - (drone.get_image_angle() / 2.0))
            b = a * np.sin(C) / np.sin(B)
            c = a * np.sin(A) / np.sin(B)

            image_width = c * 2.0
            image_height = drone.get_image_size()[1] * (image_width / drone.get_image_size()[0])

            areas.append((image_width * image_height, (image_width, image_height)))
        return min(areas, key=lambda t: t[0])[1]

    def compute_environment(self):
        drones = self.__drones
        points = self.__points

        # 1.- Get polygon giving a list of points
        from shapely.geometry import Polygon
        polygon = Polygon(points)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(*polygon.exterior.xy)  # Only for Python 3
        plt.savefig(Config.BASE_ROUTE + 'field_polygon.png')
        plt.clf()

        # 2.- Get minimum bounding rectangle
        # 2.1.- We need coordinates closest to south (min_x), north (max_x), west (min_y) and east (max_y)
        min_x = min(points, key=lambda t: t[0])[0]
        max_x = max(points, key=lambda t: t[0])[0]
        min_y = min(points, key=lambda t: t[1])[1]
        max_y = max(points, key=lambda t: t[1])[1]

        # 2.2.- Get number of squares verticaly (num_v) and horizontaly (num_h) giving drones' minimum image rectangle
        import math
        num_v = Config.ENVIRONMENT_ROWS
        num_h = Config.ENVIRONMENT_COLUMNS

        # 3.3.- Create a numpy matrix with a cell for each image square
        import numpy as np
        environment = np.zeros((num_h, num_v))

        # 3.4.- Get coordinates deltas for computing points
        d_v = (max_y - min_y) / num_v
        d_h = (max_x - min_x) / num_h

        # 3.4 Get original operator's point
        from shapely.ops import nearest_points
        closest_point = nearest_points(polygon.exterior, self.__operator_position)[0]

        # 3.5.- Check visitable squares as 1
        import itertools
        for (i, j) in itertools.product(list(range(num_v)), list(range(num_h))):  # i: [0, num_v-1], j: [0, num_h-1]
            sp1 = (j * d_h + min_x, (num_v - i) * d_v + min_y)
            sp2 = ((j + 1) * d_h + min_x, (num_v - i) * d_v + min_y)
            sp3 = (j * d_h + min_x, (num_v - (i + 1)) * d_v + min_y)
            sp4 = ((j + 1) * d_h + min_x, (num_v - (i + 1)) * d_v + min_y)
            square = Polygon([sp1, sp2, sp4, sp3])

            if Config.SQUARE:
                environment[num_h - (j + 1), num_v - (i + 1)] = 1.0  # Marked as navigable square

            if polygon.intersects(square.buffer(1e-9)) or polygon.contains(square.buffer(1e-9)):
               
                if not Config.SQUARE:
                    environment[num_h - (j + 1), num_v - (i + 1)] = 1.0  # Marked as navigable square

                if Config.START_CORNER_0_0 and Config.SQUARE:
                    self.__drone_initial_position = (0, 0)
                elif closest_point.within(square) or closest_point.intersects(square):
                    self.__drone_initial_position = (
                        num_h - (j + 1), num_v - (i + 1))  # Set operator's position as initial position

        self.__original_environment = environment

        import numpy as np
        np.savetxt(Config.BASE_ROUTE + Config.MAP_ROUTE, environment)

        import matplotlib
        matplotlib.use('Agg')  # For running in SO without graphical environment
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        computed_environment = environment.copy()
        computed_environment[self.__drone_initial_position] = 3
        ax.pcolor(computed_environment, cmap='Greys', edgecolors='gray')
        plt.savefig(Config.BASE_ROUTE + 'computed_environment.png')
        plt.clf()

        return environment

    def reset(self):
        # Reset environment and agents position
        for drone_number in range(len(self.__drones)):
            self.__agents[drone_number].set_position(self.__drone_initial_position)
            self.__agents[drone_number].reset_movements()
            self.__agents[drone_number].set_actions_taken(0)
            self.__agents[drone_number].set_valid_taken_actions(0)

    def save_agents_movements(self, done_count):
        for agent in self.__agents:
            with open(Config.BASE_ROUTE + 'drones_movement_agent' + str(agent.get_number() * 10) + '_done' + str(
                    done_count) + '_.txt',
                      'w') as f:
                f.write(agent.get_name() + ", value %s: " % movement)
                for movement in agent.get_movements:
                    f.write("%s, " % movement)
                f.write('\n')

    def compute_path(self):
        count = 0
        from Agent.Agent import Agent
        self.__agents = []
        for drone in program.get_drones():  # Create Reinforcement Learning Agents
            self.__agents.append(
                Agent(drone.get_name(), count, drone.get_battery_time(),
                      drone.get_speed(),
                      program.compute_minimum_area(self.__drones), (0, 0), self.__original_environment))

            count += 1

        # Get number of observation episodes
        number_episodes = Config.SIMULATIONS

        import time
        global_execution_start_time = time.time()
        start_number = 0
        done_count = 0  # Number of times problem has been solved

        # Get epsilon
        epsilon = Config.EPSILON

        # Save epsilon for plotting
        epsilons = [epsilon]

        # Total repetitions in all episodes
        total_unchanged_environment_episodes_count = 0

        # Maximum coverage overall
        max_coverage = 0.0

        # Max coverage lists for plotting for the whole experiment
        max_coverages = []

        # Simulations' times
        episodes_time = []

        # Simulations' total rewards
        rewards_episodes = []

        # Store total actions taken per observation
        episode_total_actions = []
        episode_total_valid_actions = []

        valid_actions_taken_agent = []

        # Compute episodes
        for episode_number in range(start_number, number_episodes):

            # Reset agents and environment
            program.reset()

            # Update heatmap
            heatmap = self.get_environment() * 0.0
            for element in self.__agents:
                (x, y) = element.get_position()
                heatmap[x][y] += 1.0

            # Add minimum max coverage
            max_coverages.append(0.0)

            # Add max coverage observation
            coverages_episode = [0.0]

            # Reset unchanged environments count
            unchanged_environment_episodes_count = 0

            # Create ANN if necessary
            if (Config.GLOBAL_MODEL):
                from numpy import dstack
                input_matrix = dstack((self.get_environment(), self.get_environment(), self.get_environment()))
                from Model.Model import create_model
                model = create_model(input_matrix.shape)

            # Get initial environment for starting observation
            actual_environment = program.get_environment()

            # Get visited positions map and agent position map
            import numpy as np
            actual_visited_map = np.array(actual_environment * 0.0, dtype=bool)  # Changed to bool for first experiments
            drone_map = np.array(actual_environment * 0.0, dtype=bool)  # Changed to bool for first experiments

            # Rewards and for plotting
            rewards_episodes.append(0.0)
            rewards = []
            action_rewards = []
            for _ in self.__agents:
                rewards.append([0])
                action_rewards.append([0])

            # Mark agents positions as true
            for agent in self.__agents:
                (i, j) = agent.get_position()
                drone_map[i, j] = True
                actual_visited_map[i, j] = True

            # Print trace every 100 episodes
            if episode_number % Config.SIMULATIONS_CHECKPOINT == 0 and Config.PRINT_SIMULATIONS:
                print("Episode {} of {}".format(episode_number + 1, number_episodes))

            # Compute paths
            done = False
            episode_counter = 0
            visited_list = []  # store each agent's visited squares
            visited_list.append(actual_visited_map)  # store each agent's visited squares

            # Add new values to actions lists
            episode_total_actions.append(0.0)
            episode_total_valid_actions.append(0.0)

            if len(valid_actions_taken_agent):
                for element in self.get_agents():
                    valid_actions_taken_agent[element.get_number()].append(0.0)
            else:
                for _ in self.get_agents():
                    valid_actions_taken_agent.append([0.0])

            # Store trendline_slope
            trendline_slope = -1.0

            import time
            start_time = time.time()
            while not done:

                # Get previous environment (this way all agents would act at the same time)
                prev_visited_map = np.array(np.ceil(np.sum(visited_list, axis=0)), dtype=bool).copy()
                prev_drone_map = drone_map.copy()
                drone_position_list = []  # store each agent's position

                # For each agent compute 1 action
                for agent in program.get_agents():

                    # Make decision
                    import numpy as np
                    rand_number = np.random.random()

                    if rand_number < epsilon:
                        random_action = True
                        # Get random action
                        chosen_action = np.random.randint(0, len(Config.ACTIONS_DICT.keys()))
                    else:
                        random_action = False
                        # Decide one action
                        if not Config.GLOBAL_MODEL:
                            chosen_action = np.argmax(agent.predict(np.array(prev_visited_map, dtype=int),
                                                                    np.array(prev_drone_map, dtype=int),
                                                                    self.get_environment(), ))
                        else:
                            chosen_action = np.argmax(agent.predict_global_model(np.array(prev_visited_map, dtype=int),
                                                                                 np.array(prev_drone_map, dtype=int),
                                                                                 self.get_environment(),
                                                                                 model))

                    episode_total_actions[episode_number] += 1.0

                    # Get agent's position before doing action for printing it in a file
                    prev_position = agent.get_position()

                    # Update environment according to action
                    actual_visited_map, actual_drone_map, reward = agent.do_action(chosen_action,
                                                                                   self.__original_environment,
                                                                                   prev_visited_map, prev_drone_map)

                    (r, c) = agent.get_position()
                    heatmap[r][c] += 1.0

                    # Plot heatmap
                    import matplotlib
                    matplotlib.use('Agg')  # For running in SO without graphical environment
                    import matplotlib.pyplot as plt
                    plt.plot(rewards[agent.get_number()])
                    fig, ax = plt.subplots()
                    im = ax.imshow(heatmap)
                    for r in range(Config.ENVIRONMENT_ROWS):
                        for c in range(Config.ENVIRONMENT_COLUMNS):
                            text = ax.text(c, r, heatmap[r, c], ha="center", va="center", color="w")
                    fig.tight_layout()
                    plt.savefig('heatmap_episode_' + str(episode_number) + '.png')
                    plt.clf()

                    # Plot agent's reward graph
                    from numpy import sum
                    rewards[agent.get_number()].append(sum(rewards[agent.get_number()]) + agent.get_reward())
                    action_rewards[agent.get_number()].append(agent.get_reward())
                    rewards_episodes[episode_number] += agent.get_reward()
                    import matplotlib
                    matplotlib.use('Agg')  # For running in SO without graphical environment
                    import matplotlib.pyplot as plt
                    plt.plot(rewards[agent.get_number()])
                    plt.savefig('total_reward_evolution_drone_' + str(agent.get_number()) + '.png')
                    plt.clf()
                    plt.plot(action_rewards[agent.get_number()])
                    plt.savefig('action_reward_evolution_drone_' + str(agent.get_number()) + '.png')
                    plt.clf()

                    if (prev_visited_map != actual_visited_map).any():
                        agent.increase_valid_taken_actions()
                        episode_total_valid_actions[episode_number] += 1.0

                    # Store the number of times in a row that the environment does not change
                    if (prev_visited_map == actual_visited_map).all():
                        unchanged_environment_episodes_count += 1
                    else:
                        unchanged_environment_episodes_count = 0

                    # Save taken action in a file
                    with open(
                            Config.BASE_ROUTE + 'actions_' + str(agent.get_number()) + '_' + agent.get_name() + '.csv',
                            'a+') as f:
                        if not episode_counter:
                            agent.set_status('flying')
                            f.write(
                                'action_code, action_name, prev_position, actual_position, valid, visited, random_action, environment_shape, actions_taken, valid_taken_actions, unchanged_episodes\n')
                        f.write(str(chosen_action) + ', ' + Config.ACTIONS_DICT[chosen_action] + ', ' + str(
                            prev_position) + ', ' + str(agent.get_position()) + ', ' + str(
                            prev_position != agent.get_position())
                                + ', ' + str((prev_position != agent.get_position()) and
                                             (prev_visited_map[agent.get_position()[0], agent.get_position()[1]]))
                                + ', ' + str(random_action)
                                + ', ' + str(self.__original_environment.shape) + ', ' + str(agent.get_actions_taken())
                                + ', ' + str(agent.get_valid_taken_actions()) + ', ' + str(
                            unchanged_environment_episodes_count) + '\n')

                    # Memorize new memory observation
                    observation = (
                    prev_visited_map, actual_visited_map, prev_drone_map, actual_drone_map, chosen_action,
                    reward, agent.get_status())
                    agent.memorize(observation)

                    # Save agent results for merging with the remaining agents
                    visited_list.append(actual_visited_map + (1.0 - self.get_environment()))
                    import matplotlib
                    matplotlib.use('Agg')  # For running in SO without graphical environment
                    import matplotlib.pyplot as plt
                    plt.imshow(np.array(np.ceil(np.sum(visited_list, axis=0)), dtype=bool), cmap='Greys',
                               interpolation='nearest')
                    plt.savefig(Config.BASE_ROUTE + 'combined_visited_list.png')
                    plt.clf()

                    drone_position_list.append(actual_drone_map)

                    # Train
                    if not Config.GLOBAL_MODEL:
                        agent_history = agent.learn(self.get_environment())
                        agent.get_model().save(str(agent.get_number()) + '_local_model.h5')
                    else:
                        agent_history = agent.learn_global_model(self.get_environment(), model)
                        model.save('global_model.h5')

                    # Check experiment stopping
                    waiting_hours = float(time.time() - start_time) / 60.0 / 60.0  # Convert seconds to hours

                    import numpy as np
                    borders_matrix = 1.0 - np.ceil(self.get_environment())
                    visited_matrix = np.array(np.ceil(np.sum(visited_list, axis=0)), dtype=float)
                    visited_matrix = np.where(visited_matrix >= 1.0, 1.0, visited_matrix)
                    only_visited_cells_matrix = visited_matrix - borders_matrix

                    visited_cells_count = float(np.count_nonzero(only_visited_cells_matrix == 1.0))
                    visitable_cells_count = float(np.count_nonzero(self.get_environment() == 1.0))
                    coverage = visited_cells_count / visitable_cells_count

                    max_coverage = max(coverage, max_coverage)
                    max_coverages[episode_number] = max(coverage, max_coverages[episode_number])
                    coverages_episode.append(coverage)

                    valid_actions_taken_agent[agent.get_number()][episode_number] = agent.get_valid_taken_actions()

                    if unchanged_environment_episodes_count >= Config.MAXIMUM_UNCHANGED_ENVIRONMENT_EPISODES:
                        total_unchanged_environment_episodes_count += unchanged_environment_episodes_count
                        done = True
                        break
                    elif waiting_hours >= Config.MAXIMUM_WAIT_HOURS and coverage < Config.COMPLETENESS_COVERAGE:
                        total_unchanged_environment_episodes_count += unchanged_environment_episodes_count
                        done = True
                        break

                    # Check if agent had finished
                    if False not in np.array(np.ceil(np.sum(visited_list, axis=0)), dtype=bool):
                        with open(Config.BASE_ROUTE + 'solution_times.txt', 'a+') as f:
                            f.write('solution time ' + str(done_count) + ': '
                                    + time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
                                    + ' epsilon: ' + str(epsilon)
                                    + '\n')
                        done_count += 1
                        done = True
                        break

                episode_counter += 1

                # Combine agents results
                drone_map = np.array(np.sum(drone_position_list, axis=0), dtype=bool)

            # Plot coverages for each observation graph
            if len(coverages_episode) > 1:
                import matplotlib
                matplotlib.use('Agg')  # For running in SO without graphical environment
                import matplotlib.pyplot as plt
                ax = plt.figure().gca()
                ax.set_ylim([0.0, 1.0])
                x = list(range(len(coverages_episode)))
                y = coverages_episode
                from numpy import polyfit
                fit = polyfit(x, y, 1)
                yfit = [n * fit[0] for n in x] + fit[1]
                ax.plot(x, y)
                ax.plot(yfit, 'r--')
                plt.savefig('coverages_episode_' + str(episode_number) + '.png')
                plt.clf()

            # Store and plot observation's time
            episodes_time.append((time.time() - start_time) / 3600.0)
            import numpy as np
            average_episode_time = np.average(episodes_time)
            import matplotlib
            matplotlib.use('Agg')  # For running in SO without graphical environment
            import matplotlib.pyplot as plt
            ax = plt.figure().gca()
            ax.plot(episodes_time)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig('episode_time_hours.png')
            plt.clf()


            # Plot valid action percentage per observation graph
            if len(episode_total_valid_actions) > 1:
                import matplotlib
                matplotlib.use('Agg')  # For running in SO without graphical environment
                import matplotlib.pyplot as plt
                import numpy as np
                ax = plt.figure().gca()
                division = np.divide(episode_total_valid_actions, episode_total_actions)
                ax.set_ylim([0.0, 1.0])
                x = list(range(len(division)))
                y = division
                from numpy import polyfit
                fit = polyfit(x, y, 1)
                yfit = [n * fit[0] for n in x] + fit[1]
                ax.plot(x, y)
                ax.plot(yfit, 'r--')
                plt.savefig('actions_percentages_episodes.png')
                plt.clf()

                import matplotlib
                matplotlib.use('Agg')  # For running in SO without graphical environment
                import matplotlib.pyplot as plt
                import numpy as np
                ax = plt.figure().gca()
                ax.set_ylim([0.0, 1.0])
                for element in self.get_agents():
                    division = np.divide(valid_actions_taken_agent[element.get_number()], episode_total_actions)
                    x = list(range(len(division)))
                    y = division
                    ax.plot(x, y)
                plt.savefig('percentage_work_per_agent.png')
                plt.clf()

            # Plot coverages graph
            if len(max_coverages) > 1:
                import matplotlib
                matplotlib.use('Agg')  # For running in SO without graphical environment
                import matplotlib.pyplot as plt
                ax = plt.figure().gca()
                ax.set_ylim(bottom=0.0)
                x = list(range(len(max_coverages)))
                y = max_coverages
                from scipy.stats import linregress
                trend = linregress(x, y)
                trendline_slope = trend.slope  # or fit[0]
                from numpy import polyfit
                fit = polyfit(x, y, 1)
                yfit = [n * fit[0] for n in x] + fit[1]
                ax.plot(x, y)
                ax.plot(yfit, 'r--')
                plt.savefig('coverages.png')
                plt.clf()

            # Plot epsilon graph
            import matplotlib
            matplotlib.use('Agg')  # For running in SO without graphical environment
            import matplotlib.pyplot as plt
            ax = plt.figure().gca()
            ax.plot(epsilons)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig('epsilons.png')
            plt.clf()

            # Update epsilon
            # The lower the epsilon, less random actions are taken
            epsilon = max(Config.MIN_EPSILON, epsilon * Config.EPSILON_DECAY)
            epsilons.append(epsilon)

def specify_random_seed():
    import numpy as np

    if Config.SEED == None:
        # Get random seed
        Config.SEED = np.random.randint(1, 255)

            # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(Config.SEED)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(Config.SEED)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(Config.SEED)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    if tf.__version__ < '2.0.0':
        tf.set_random_seed(Config.SEED)
    else:
        import tensorflow.compat.v1 as tf
        tf.set_random_seed(Config.SEED)

    # 5. Configure a new global `tensorflow` session
    # if tf.__version__ >= '2.0.0':
    #    import tensorflow.compat.v1 as tf
    #    tf.disable_v2_behavior()
    # import tensorflow.python.keras.backend as K
    from tensorflow.python.keras import backend as K
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # 6. Save seed to a file
    with open(Config.BASE_ROUTE + 'session_seed.txt', 'w') as seed_file:
        seed_file.write(str(Config.SEED) + '\n')
        seed_file.close()


def run_tests():
    import unittest
    from Tests.ProgramTests import ProgramTests
    test_classes_to_run = [ProgramTests]
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    result = runner.run(big_suite)

    print('Tests run ', result.testsRun)
    print('Error number: ', len(result.errors))
    print('Errors ', result.errors)
    print('Failure number: ', len(result.failures))
    print('Failures ', result.failures)


if __name__ == '__main__':
    if Config.UNIT_TESTS:
        print('\n\n\nRun unit tests')
        run_tests()

    print('\n\n\nSetting random seed')
    specify_random_seed()

    print('\n\n\nInitializing program')
    program = Program()

    print('\n\n\nReading configuration')
    program.read_data()

    print('\n\n\nCompute flying environment')
    if not Config.LOAD_MAP_FILE:
        program.compute_environment()
    else:
        import numpy as np
        program.set_environment(np.loadtxt(Config.BASE_ROUTE + Config.MAP_ROUTE))

    print('\n\n\nCompute flying path')
    program.compute_path()
