
"""
Learning the Q-Table
So that's pretty simple, right? Now how do we create this table and find those values. Well this where we will discuss how the Q-Learning
Algorithm updates the values in our Q-Table.

I'll start by noting the our Q-Table starts of with all 0 Values. This is because the agent has yet to learn anything about the enivornment.
Our agent learns by exploring the environment and observing the outcome/reward from each action it takes in a given state. But how does it know 
what action to take in each state? There are two ways that our agent can decide on which action to take.

  1. Randomly picking a valid action
  2. Using the current Q-Table to find the best action
  
Near the beginning of our agents learning it will mostly take random actions in order to explore the environment and enter many different states.
As it starts to explore more of the environment it will start to gradually rely more on it's learned values (Q-Table) to take actions. 

This means that as our agent explores more of the environment it will develop a better understanding and start to take "correct" or better actions 
more often. It's important that the agent has a good balance of taking random actions and using learned values to ensure it does get trapped in a local maxima.

After each new action our agent will record the new state(if any) that it has entered and the reward that it recieved from taking that action. These values 
will be used to update the Q-Table. The agent will stop taking new actions only once a certain time limit is reached or it has achieved the goal or reached the
end of the environment.

Updating the Q-Values:
The formula for updating the Q-Table after each action is as follows:

>>>>>  Q[state, action] = Q [state, action] + α * (reward + γ * max(Q[newState, :]) - Q[state, action])
>>>>>>>>  α stands for the Learning Rate 
>>>>>>>>  γ stands for the Discount Factor

Learning Rate α:
This learning rate α is a numeric constant that defines how much change is permitted on each QTable update. A high learning rate means that each update will introduce a large change to the 
curent state-action value. A small learning rate means that each update will introduce a large change to the ccurrent state-action value. A small learning rate means that each update has a
more sublte change. Modifying the learning rate the learning rate will change how the agent explores the environment and how quickly it determines the final values inthe QTTable.

Discount Factor  γ :
Discount Factor, also know as gamma ( γ ) is used to balance how much focus is put on the current and future reward.A high discount factor means will be considered more heavily.
To perform updates on this table we will let the agent explore the environment for a certain period

"""


""" 
From Tutorial:
So we're gonna now talk about learning the Q table. So essentially, I showed you how we use that Q table, which is given some state, we just look 
that state up in the Q table, and then determine what the maximum reward we could get by taking you know, some actions and then take that action. And
that's how we would use the Q table latter on when we're actually using the model. 

But when we're learninf the Q table, that's not necessarily what we want to do, we don't want to explore the environment by just taking the maximum reward 
that we've seen so far. And just always going the direction, we need to make sure that we're exploring in a different way and learning the correct values for the Q Table. 

So essentially, our agent learns by exploring the environment and observing the outcome slash reward from each action it takes in a given state, which we've
already set. But how does it know what action to take in each state when it's learning? That's the question I need to answer for you now, well, there's two ways of doing this,
our agent can essentially, you know, use the current Q table to find the best action which is kind of what I just discussed. 

So taking, looking at the Q table, looking at the state and just taking the highest reward or it can randomly pick a valid action. And our goal is going to be when we create this
Q Learning Algorithm to have a really great balance of these two, where sometimes we use the Q table to find the best action, and sometimes we take a random action. So this is 
one thing. But now I'm just to talk about this formula for how we actually update Q values. So obviously, what's gonna end up happening in our Q learning is, we're gonna have an
agent, that's going to be in the learning stage, exploring the environment and having all these actions and all these rewards and all these observations happening. 

And it's going to be moving around the environment by following one of these two kind of principles, randomly picking a valid action or using the current Q table to find the best action.
When it gets into a new state. And it you know, moves from state to state, it's going to keep updating the Q Table telling it, you know this is what I've learned about the environment, 
I think this is a better move, we're going to update this value. 

But how does it do that? In a way that's going to make sense because we cannot just put, you know the maximum way, that's going to make sense because we cannot just put, you know, 
the maximum value we got for moving otherwise, we're going to run into that issue, which I just talked about, where we get stuck in that local maxima , right? I am not sure if I called it
minimum before. But anyways, it's local maxima where we see this high reward.

But that's preventing us if we keep taking that action from reaching a potentially high reward in a different state, so the formula that we actually use to update the Q table is this.
So the formula is 
>>>>>  Q[state, action] = Q [state, action] + α * (reward + γ * max(Q[newState, :]) - Q[state, action])
>>>>>>>>  α stands for the Learning Rate 
>>>>>>>>  γ stands for the Discount Factor

So what is the heck is above equations and Constants? What is all this? We're going to talk about the constants in a minute, but I want to yeah, I want to explain this formula, actually.
So let's okay, well I guess we'll go through the constants, it's hard to go through a complicated MATH FORMULA. So there are two symbols aplha for learning and gamma for discount factor.




So what is learning rate? Essentially rate ensures that we don't update our Q table too much on every observation. So before, right when I was showing you from the table diagram look at the 
diagrammatic table when I took an action was I looked at the reward that I got from taking that action. And I just put that in my Q table right? Now, obviously, that is not an optimal appoarch
to do this, because that means that in the instance, where we hit the state 1, well , I'm not going to be able to get this reward of 4, because I'M going to throw that you know, 3 in here and I'm
just going to keep taking that action. 

"We need to, you know, hopefully make this move action actually have a higher value than stay. " So that next time we're in state1, we consider the fact that next time, we're in state1, we consider
the fact that we can move to state2, and then move to state3 to optimize a reward. So how do we do that? Well, the learning rate is one thing that helps us kind of accomplish this behaviour. Essentially,
what it is telling us and this is usually a decimal value, right, is how much we're allowed to update every single cue value by on every single action or every single observatino.

So if we just use the approach before, then we're only going to need to observe, given the amount of states and the amount of actions and we'll be able to completely fill inthe Q Table. So in our case, if 
we had like three states,and three actions, we could you know, nine iterations we'd be able to fill the entire Q table, the learning rate means that it's going to  just update a little bit slower, and 
essentially, change the value in the Q table very slightly.

So you can see that what we're doing is taking the current value of the Q Table, so whatever is already there, and then what we're going to do is add some value here. And this value that we add is either
going to be positive or negative essentially telling us you know, whether we should take this new "action" or whether we shoudlnot take this new action. Now, the way that this formula kind of value is 
calculated, right, is obviously of our alpha is multiplied this by this, but we have the reward, plus, in this case, gamma , which is just going to actually be the discount factor. And I 'll talk about
how that works in a second maximum of the new state we moved into. 

Now, what this means is find the maximum reward that we could receive in the new state by taking any action and multiply that by what we call the discount factor. What this part of formula is trying 
to do is exactly what I've kind of been talking about, try to look forward and say, okay, so I know if I take this action in this state, I receive this amount of reward, but I need to factor in the reward
I could receive in the next state, so that I can determine the best place to move to. That's kind of what this Max and this gamma are trying to do for us. So in this discount factor, whatever you want
to call it, it's trying to factor in a little bit about what we could get from the next state into this equation.

So that hopefully, our kind of agent can learn a little bit more about the transition states. So states that maybe are actions that maybe don't give us an immediate reward, but lead to a larger reward in the 
future, that's what this γ * max(Q[newState, :]) are trying to do. Then what we do is we subtract from this the state and action - Q[state, action]) just to make sure that we're adding what the difference
was in know, what we get from this versus what the current value is and not like mutliplying these values, crazily. I mean, you can look into more of the math here and plug in like some 
values later,

AND YOU'LL SEE THIS KIND of works But this is the basic format. I feel like I explained that in depth enough. OKay, so now that we'vwe done that , and we've updated this, we've learned kind of how we update the 
cells and how this works and I could go back to the whiteboard and draw it our. But i feel like that makes enough sense, we 're going to look at what the next state is , we're going to factor that into our
calculation tells us eseentially how much we  canupdate each cell value by and we have this, what do you call it here discount factor, which essentially tries to kind of defined the balance between finding really
good rewards in our current state and finding the rewards in the future state.so the higher this value is the more we 're going to look towards the future. The lower it is, the more we're going to focus completely
on our current reward. 

Right, and obviously that makes sense because we're going to add the maximum value. and if we're multiplying that by a lower number that means we are going to consider that less than if that 
was greater Awesome okay.
"""

"""
Algorithms: 
1. Create 3 state table S1, S2, AND S3 and the values for s1 = 1, s2 =2 and s3 = 3
2. S1 to s2 is 5 and s2 to s3 is 8 and s3 to s1 is 0 
    and s2 to s1 is 3, s3 to s2 is 5, and s1 to s3 is 2
3. now compute table for above connections with the tables   
4. now starting state is s1 and choose the highest reward to connect states
5. then maintain the new with the connections and paths table
"""
class Agent:
    def __init__(self):
        self.states = {'s1': 1, 's2': 2, 's3': 3}
        self.connections = {'s1': {'s2': 5, 's3': 2},
                            's2': {'s1': 1, 's3': 8},
                            's3': {'s1': 100, 's2': 5}}
        self.paths = {}

    def compute_highest_reward_transitions(self):
        for state in self.states:
            max_reward = 0
            next_state = None
            for next_state_candidate, reward in self.connections[state].items():
                if reward > max_reward:
                    max_reward = reward
                    next_state = next_state_candidate
            self.paths[state] = next_state

    def update_table(self):
        print("State\tValue")
        for state, value in self.states.items():
            print(f"{state}\t{value}")

    def update_connections(self):
        print("\Rewards:")
        for from_state, transitions in self.connections.items():
            for to_state, reward in transitions.items():
                print(f"{from_state} -> {to_state}: {reward}")

    def update_paths(self):
        print("\nActions of Agent:")
        for state, next_state in self.paths.items():
            print(f"{state} -> {next_state}")
            
if __name__ == "__main__":
    agent = Agent()
    
    # Compute highest reward transitions
    agent.compute_highest_reward_transitions()

    # Update and print the state table, connections, and paths
    agent.update_table()
    agent.update_connections()
    agent.update_paths()
