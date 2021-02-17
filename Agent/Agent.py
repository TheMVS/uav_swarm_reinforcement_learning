# -*- coding: utf-8 -*-
import Config


class Agent:

    def __init__(self, name, number, autonomy_time, speed, minimum_image_size, position, environment_matrix):
        self.__status = 'start'
        self.__model = None
        if not Config.GLOBAL_MODEL:
            import numpy as np
            input_matrix = np.dstack((environment_matrix, environment_matrix, environment_matrix))
            from Model.Model import create_model
            self.__model = create_model(input_matrix.shape)
        self.__number = number  # maybe it is better to create random color: tuple(np.random.randint(256, size=3))
        self.__name = name
        self.__autonomy_time = autonomy_time
        self.__time_move_region_horizontal = float(minimum_image_size[0]) / float(speed)
        self.__time_move_region_vertical = float(minimum_image_size[1]) / float(speed)
        self.__position = position
        self.__movements = []
        self.__valid_actions = []
        self.__memory = []
        self.__reward = 0
        self.__cycle_count = 0
        self.__actions_taken = 0
        self.__valid_taken_actions = 0

    # Getters and setters
    def get_name(self):
        return self.__name

    def get_status(self):
        return self.__status

    def get_time_move_region_lateral(self):
        return self.__time_move_region_horizontal

    def get_time_move_region_vertical(self):
        return self.__time_move_region_vertical

    def get_autonomy_time(self):
        return self.__autonomy_time

    def get_number(self):
        return self.__number

    def get_reward(self):
        return self.__reward

    def get_position(self):
        return self.__position

    def set_position(self, new_position):
        self.__position = new_position

    def set_movements(self, movement_list):
        self.__movements = movement_list

    def set_status(self, status):
        self.__status

    def get_model(self):
        return self.__model

    def get_actions_taken(self):
        return self.__actions_taken

    def set_actions_taken(self, number):
        self.__actions_taken = number

    def get_valid_taken_actions(self):
        return self.__valid_taken_actions

    def set_valid_taken_actions(self, number):
        self.__valid_taken_actions = number

    def increase_valid_taken_actions(self):
        self.__valid_taken_actions += 1

    # Other methods
    def reset_movements(self):
        self.__movements = []

    def decrease_autonomy_time(self, amount):
        self.__autonomy_time -= amount

    def compute_valid_actions(self, environment):
        # Get total number of columns
        (rows, columns) = environment.shape

        # Get actual position
        (row, column) = self.__position

        actions = []  # Possible actions list
        import numpy as np
        if column - 1 >= 0 and np.round(environment[row, column - 1]) > 0:
            # left
            actions.append(0)
        if row - 1 >= 0 and np.round(environment[row - 1, column]) > 0:
            # up
            actions.append(1)
        if column + 1 < columns and np.round(environment[row, column + 1]) > 0:
            # right
            actions.append(2)
        if row + 1 < rows and np.round(environment[row + 1, column]) > 0:
            # down
            actions.append(3)

        self.__valid_actions = actions

    def do_action(self, chosen_action, environment, prev_visited_map, prev_agent_map):
        # Update number of actions taken
        self.__actions_taken += 1

        # Copy values for updating
        new_agent_map = prev_agent_map.copy()
        new_visited_map = prev_visited_map.copy()

        # Get valid actions agent can do
        self.compute_valid_actions(environment)

        # Update environment and agent's position
        prev_autonomy = self.__autonomy_time
        if chosen_action in self.__valid_actions:
            new_agent_map[self.__position[0], self.__position[1]] = False

            # Store all valid chosen movements for printing it
            self.__movements.append(Config.ACTIONS_DICT[chosen_action])

            (row, col) = self.__position
            if chosen_action == 0:  # left
                (row, col) = (row, col - 1)
            if chosen_action == 1:  # up
                (row, col) = (row - 1, col)
            if chosen_action == 2:  # right
                (row, col) = (row, col + 1)
            if chosen_action == 3:  # down
                (row, col) = (row + 1, col)
            self.__position = (row, col)
            # environment[row, col] = self.__number
            if prev_visited_map[row, col]:
                self.__cycle_count += 1
                self.__reward = Config.VISITED_CELL_REWARD - float(self.__cycle_count) / float(Config.ENVIRONMENT_ROWS * Config.ENVIRONMENT_COLUMNS)
            else:
                import numpy as np
                # Reward is increased when there are less remaining new cells and added random component in order to
                # emulate Skinners variable rewards, so each UAV will have different rewards for new cells and it is
                # less possible to go to the same cell as another uav in the same iteraction
                self.__reward = Config.NEW_CELL_REWARD * (1.0 + max(Config.ENVIRONMENT_ROWS,
                                Config.ENVIRONMENT_COLUMNS) / np.count_nonzero(
                                new_visited_map == False))

            new_visited_map[row, col] = True  # In future asign it to agent's number
            new_agent_map[row, col] = True

        else:
            self.__reward = Config.NO_CELL_REWARD

        # Return tuple: (updated environment, penalization)
        return new_visited_map, new_agent_map, self.__reward

    def memorize(self, observation):
        # Update agent memory
        import Config
        self.__memory.append(observation)
        while len(self.__memory) > Config.MEMORY_SIZE:  # Forget old data if memory is full
            self.__memory.pop(0)

    # --------------- Multiple models ---------------

    def prepare_data(self, environment):
        # Prepare data for training
        import Config
        import numpy as np
        aux = np.array([(self.__memory[0][0], self.__memory[0][0])])
        env_size = aux.shape
        data_size = Config.MEMORY_SIZE
        mem_size = len(self.__memory)  # in case we didn't use all memory
        data_size = min(mem_size, data_size)
        inputs1 = []
        inputs2 = []
        outputs = np.zeros((data_size, len(Config.ACTIONS_DICT.keys())),
                           dtype=float)  # We have len(Config.ACTIONS_DICT.keys() actions

        # For each observation at memory
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            # i = ordered memory position, j = random memory position ===> more generalization if ANN is
            # trained without order

            # Get observation i from memory
            prev_visited_map, actual_visited_map, prev_agent_map, actual_agent_map, chosen_action, reward, status = \
                self.__memory[j]

            # Convert data to int for training ANN
            prev_visited_map = np.array(prev_visited_map, dtype=int)
            prev_agent_map = np.array(prev_agent_map, dtype=int)
            actual_visited_map = np.array(actual_visited_map, dtype=int)
            actual_agent_map = np.array(actual_agent_map, dtype=int)

            # Save observation i if it has not been saved before
            if len(inputs1) <= i:
                inputs1.append(prev_visited_map)  # Visited positions as inputs
                inputs2.append(prev_agent_map)  # Agents positions as inputs

            # Save target values. Theoretically, chosen/taken action will have non-zero values, remain actions'
            # values are 0. There should be no target values (0) for actions not taken.
            outputs[i] = self.predict(prev_visited_map, prev_agent_map, environment)

            # Compute max expected Q value
            predicted_q_values = self.predict(actual_visited_map, actual_agent_map, environment)
            max_q_sa = np.argmax(predicted_q_values)

            # Apply Q-function
            if status == 'finish':
                outputs[i, chosen_action] = reward
            else:
                outputs[i, chosen_action] = reward + Config.GAMMA * max_q_sa
        return inputs1, inputs2, outputs

    def learn(self, environment):
        import Config
        inputs1, inputs2, outputs = self.prepare_data(environment)
        ann_input = []
        for t in zip(inputs1, inputs2):  # For each observation
            import numpy as np
            r, c = t[0].shape
            ann_input.append(np.dstack((t[0], t[1], np.zeros((r, c), float))))  # 3D matrix with data

        # Fit ANN
        ann_input = np.asarray(ann_input)
        history = self.__model.fit([ann_input], outputs, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
                                   verbose=Config.VERBOSE)
        return history

    def predict(self, environment1, environment2, environment3):
        import numpy as np
        ann_input = np.dstack((environment1, environment2, environment3))  # 3D matrix with data

        # Predict Q-table values
        return self.__model.predict(np.array([ann_input]))[0]  # [0] only with Keras

    # --------------- Global Model ---------------

    def learn_global_model(self, environment, model):
        import Config
        inputs1, inputs2, outputs = self.prepare_data_global_model(environment, model)
        ann_input = []
        for t in zip(inputs1, inputs2):  # For each observation
            import numpy as np
            r, c = t[0].shape
            ann_input.append(np.dstack((t[0], t[1], np.zeros((r, c), float))))  # 3D matrix with data

        # Fit ANN
        history = model.fit([ann_input], outputs, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
                            verbose=Config.VERBOSE)
        return history

    def prepare_data_global_model(self, environment, model):
        # Prepare data for training
        import Config
        import numpy as np
        aux = np.array([(self.__memory[0][0], self.__memory[0][0])])
        env_size = aux.shape
        data_size = Config.MEMORY_SIZE
        mem_size = len(self.__memory)  # in case we didn't use all memory
        data_size = min(mem_size, data_size)
        inputs1 = []
        inputs2 = []
        outputs = np.zeros((data_size, len(Config.ACTIONS_DICT.keys())),
                           dtype=float)  # We have len(Config.ACTIONS_DICT.keys() actions

        # For each observation at memory
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            # i = ordered memory position, j = random memory position ===> more generalization if ANN is
            # trained without order

            # Get observation i from memory
            prev_visited_map, actual_visited_map, prev_agent_map, actual_agent_map, chosen_action, reward, status = \
                self.__memory[j]

            # Convert data to int for training ANN
            prev_visited_map = np.array(prev_visited_map, dtype=int)
            prev_agent_map = np.array(prev_agent_map, dtype=int)
            actual_visited_map = np.array(actual_visited_map, dtype=int)
            actual_agent_map = np.array(actual_agent_map, dtype=int)

            # Save observation i if it has not been saved before
            if len(inputs1) <= i:
                inputs1.append(prev_visited_map)  # Visited positions as inputs
                inputs2.append(prev_agent_map)  # Agents positions as inputs

            # Save target values. Theoretically, chosen/taken action will have non-zero values, remain actions'
            # values are 0. There should be no target values (0) for actions not taken.
            outputs[i] = self.predict_global_model(prev_visited_map, prev_agent_map, environment, model)

            # Compute max expected Q value
            predicted_q_values = self.predict_global_model(actual_visited_map, actual_agent_map, environment, model)
            max_q_sa = np.argmax(predicted_q_values)

            # Apply Q-function
            if status == 'finish':
                outputs[i, chosen_action] = reward
            else:
                outputs[i, chosen_action] = reward + Config.GAMMA * max_q_sa
        return inputs1, inputs2, outputs

    def predict_global_model(self, environment1, environment2, environment3, model):
        import numpy as np
        r, c = environment1.shape
        ann_input = np.dstack((environment1, environment2, environment3))  # 3D matrix with data

        # Predict Q-table values
        return model.predict(np.array([ann_input]))[0]  # [0] only with Keras
