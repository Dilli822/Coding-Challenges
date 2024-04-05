
# Tutotrial: https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/reinforcement-learning-with-q-learning
"""
Reinforcement Learning - ML
The next and final topic in this course covers Reinforcement Learning. This technique is different than many of the other machine learning techinques we have seem earlier and has 
many applications in training agents (an AI) to interact with environments like games. Rather than feeding our machine learning model millions of examples we let our model come
up with it's own examples by exploring an environment. The concept is fairly simple. Humans learn by exploring and learning from mistakes and past experiences so let's have our 
computer do the same.

Terminology
Before we dive into explaining reinforcement learning we need to define a key pieces of terminology.

Environment:
In reinforcement learning tasks we have a notion of the enviornment. This is what our agent will explore. An example of an enivorment in the case of training an AI to play say a 
game of mario would be the level we are training the agent on.

Agent:
An agentis an entity that is exploring the environment, Our agent will interact and take different actions within the enviornment, In our mario example the marios characater within
the game would be our agent.

State:
At all times our agent will be in what we call a state. The state simply tells us about the status of the agent. The most common example of a state is the location of the agent within
the environment. Moving locations would change the agents state.

Action:
Any interaction between the agent and environment would be considered an action. For example moving to the left or jumping would be an action. An action may or may not change the current
state of the agent. In fact the act of doing nothing is actually an action as well! The action of say not pressing a key if we are using more mario example.

Reward:
Every action that our agent takes will result in a reward of some magnitude (positive or negative). The goal of our agent will be to maximixe it's reward in an environment, Sometimes the 
reward will be clear, for example if an agent performs an action which increases their score in the environment we could say they've recieved a positive reward. If the agent were to perform 
an action which results in them losing score or possibly dying in the environment then they would recieve a negative reward.

The most important part of reinforcement learning is determining how to reward the agent. After all, the goal of the agent is to maximize it's rewards. This means we should reward the agent
appropriatly such that it reaches the desired goal.
"""

""" 
Q-Learning:
Now that we have a vague idea of how reinforcement learning works it's time to talk about a specific technique in reinforcement learning called Q-Learning.
Q-Learning is a fairly simple yet quite powerful technique in machine learning that involves learning a matrix of action-reward values. This matrix is ofetn refered to as a  
Q-Table or Q-Matrix.
The matrix is in shape(number of possible states, number of possible actions) where each value at matrix[n, m] represents the agents expected reward given they are in state n and take action 
m. THe Q-learning algorithm defines the Q-table/matrix we can determine the action an agent should take in any state by looking at that states row in the matrix and taking the maximum value 
column as the action.

Consider this example.
Let's say A1-A4 are the possible actions and we have 3 states represented by each row (state 1 - state 3).

    A1    A2   A3    A4 
    0     0    10    5
    5     10    0    0
    10    5     0    0
If that was our Q-Table/mateix then the following would be the preffered actions in each state.
State 1: A3
State 2: A2
State 3: A1
We can see that this is because the values in each of those columns are the highest for those states.   
"""
""" 
Simple Traffic Light Signaling Agent
------- Traffic Light ----------
Psudeo Normal Algorithm
step 1: start
step 2: there are three possible states in traffic light
        green, red and yellow light
step 3: Rules are if green light then vehicles 'V' can go if it red then 'V' will stop, clear the intersection
        and people 'p' will walk out or cross the zerbra crossing or intersection

step 4: If it is Yellow 'Y' then no people will be crossing the road and it is about to appear a red light so please slow down
        the vehicles and activate the red light and repeat step 3
step 5: If not no vehicle no people then make red light


step 1: start
step 2: Create a matrix or an array table for 3 states with values between 0 to 1 in floating 
step 3: traffic sensor or matrix will get populate for each 3 states i.e.
        state 1 : red light matrix [1 1 1 1] high matrix means all vehicles must stop 
        state 2:  green light matrix [1 1 1 1] high matrix means all people must stop 
        state 3:  yellow light matrix [1 1 1 1] high means yellow appear till the matrix of red light is high [1 1 1 1]

STEPS SEQUENCE ARE RED --> GREEN --> YELLOW --> RED ---> GREEN ---> YELLO ---> CYCLIC LOOP

red light matrix = [0.3  0.4  0.5
                    0.2  0.5  0.6      this means people numbers are increasing or waiting queues for the people are increasing 
                    1.0   0.8  0.6 ]
                    
green light matrix = [0.3  0.77  0.9
                      0.65  0.5  0.7    this means vehicles numbers are increasing or vehicles numbers waiting the queues are increasing 
                      1.0   0.8  1.0 ]
if green light matrix = [1  1  1
                      1.0  1  1         as soon as this happen signal green light and then 
                      1.0   1.0  1.0 ]
yello light matrix = 
"""
import numpy as np
import time

# Create matrices for red, green, and yellow
matrix_vehicle = np.zeros((3, 3))  # Set all elements to 1 initially
matrix_people = np.ones((3,3))
light_yellow = 0.3

print(">>>>> Red Light Signal <<<<<< ")

print("Vehicles Stopping behind the Zebra Line \n", matrix_vehicle)

time.sleep(2)

print("People Numbers ->\n", matrix_people)

time.sleep(2)

# Fill the green matrix with values between 0 and 1 in a zigzag pattern
for i in range(3):
    if i % 2 == 0:
        matrix_vehicle[:, i] = np.linspace(0, 1, 3)
        matrix_people[:, i] = np.linspace(0, 0.75, 3)
    else:
        matrix_vehicle[:, i] = np.linspace(1, 0, 3)
        matrix_people[:, i] = np.linspace(0, 0.5, 3)

    print("vehicle numbers increasing and Stopping behind the zebra line:")
    print(matrix_vehicle)
    print("people crossing and decreasing numbers ")
    print(matrix_people)

# Find the indices of entries that are not equal to 1
non_one_indices = np.where(matrix_vehicle != 1)
people_zero_indices = np.where(matrix_vehicle != 0)

print("")
# Delay for 2 seconds
time.sleep(2)
print("People Crossing the Road")
# Increment the non-one entries by the difference between them and 1
matrix_vehicle[non_one_indices] += 1 - matrix_vehicle[non_one_indices]
# Find indices of non-zero entries in matrix_people
non_zero_indices = np.where(matrix_people != 0)

# Subtract each non-zero value from itself to make it zero
matrix_people[non_zero_indices] -= matrix_people[non_zero_indices]

time.sleep(2)

print("All People Crossed the Road because the matrix is ")
print(matrix_people)
print("Vehicles Matrix:")
print(matrix_vehicle)

print("\n")
print(" >>>> Green Light Signal <<<< ")
print("People are Slowly increasing the number and Stoping and Waiting Beside the Road till red light")
time.sleep(5)
print("Vehicles are Starting to Cross the Zebra crossing ")
time.sleep(5)

for i in range(3):
    if i % 3 == 0:
        matrix_vehicle[:, i] = np.round(0.5 * np.random.rand(3), 2)  # Generate and round random values between 0 and 0.5
        matrix_people[:, i] = np.round(np.random.rand(3), 2)
    else:
        matrix_people[:, i] = np.linspace(1, 0, 3)
        matrix_vehicle[:, i] = np.linspace(1, 0, 3)
    time.sleep(2)
    print("Vehicles Crossing Zebra Cross and their decreasing numbers:")
    print(matrix_vehicle)
    time.sleep(2)
    print("Vehicles Crossing Zebra Cross and the people waiting beside the road numbers increasing:")
    print(matrix_people)

print("\n")
time.sleep(4)
print("Finally Vechiles Decreasing Numbers ")
print(matrix_vehicle)
print("\n")
time.sleep(4)
print("Finally People slowly increasing and waiting Numbers ")
print(matrix_people)

# Calculate the average of all elements in matrix_red
average_vehicle_crossed = np.mean(matrix_vehicle)

print("Current Vechicles Crossing Rate ", average_vehicle_crossed)

print("Yellow Light gets activated only after the Avg vechiles crossing is less than", light_yellow)

v_close_to_cross = 0.5 - average_vehicle_crossed 
print("\n")
if(v_close_to_cross < light_yellow):
  print("Vehicles Numbers are: ", v_close_to_cross)
  print("\n")
  print(">>>>> Yellow Light. Slow Down Vehicles <<<<< ")

time.sleep(5)

print("\n")
print(">>>>> RED LIGHT <<<<<< ")
print(">>>>> Repeat RED --> GREEN ->> YELLOW --> RED <<<<<< ")
