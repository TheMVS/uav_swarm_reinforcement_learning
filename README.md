# UAV SWARM PATH PLANNING WITH REINFORCEMENT LEARNING FOR FIELD PROSPECTING: PRELIMINARY RESULTS

System for the coordination of UAV swarms for Path Planning in agricultural fields. Made in Python 3 with Open Source libraries.

Reinforcement Learning techniques are used for this system. To be more precise, it uses the Deep Q-Learning technique.

This code is necessary to experiment with the preliminary results obtained.

## Installation

To install this computer you must download this repository:

```bash
$ git clone https://github.com/TheMVS/QLearning_drone.git
```

Once downloaded, it is necessary to enter the project's root folder and install the necessary libraries with [pip](https://pip.pypa.io/en/stable/):

```bash
$ cd QLearning_drone
$ pip install -r requirements.txt
```

The requirements.txt file includes the following libraries:

 * [numpy](https://numpy.org)
 * [scipy](https://www.scipy.org)
 * [pandas](https://pandas.pydata.org)
 * [Shapely](https://shapely.readthedocs.io/en/latest/)
 * [Keras](https://keras.io)
 * [Matplotlib](https://matplotlib.org)

## Usage

### Run system

To run the system you must be in the root folder of the project and execute the file [Program.py](https://github.com/TheMVS/QLearning_drone/blob/master/Program.py):

```bash
$ cd av_swarm_learning_preliminary
$ python Program.py
```

### Configuration

All necessary data for experimented should be added to [data.json](https://github.com/TheMVS/uav_swarm_learning_preliminary/blob/main/data.json) and [Config.py](https://github.com/TheMVS/uav_swarm_learning_preliminary/blob/main/Config.py).

## Authors

* [Alejandro Puente-Castro](https://orcid.org/0000-0002-0134-6877)
* [Daniel Rivero](https://orcid.org/0000-0001-8245-3094)
* [Alejandro Pazos](https://orcid.org/0000-0003-2324-238X)
* [Enrique Fernandez-Blanco](https://orcid.org/0000-0003-3260-8734)

## License
[MIT](https://choosealicense.com/licenses/mit/)
