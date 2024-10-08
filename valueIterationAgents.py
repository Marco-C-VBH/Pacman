# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        num_iterations = self.iterations
        for i in range(num_iterations):
            value_copy = self.values.copy()
            for state in self.mdp.getStates():
                action = self.getAction(state)
                if action is not None:
                    q_value = self.getQValue(state, action)
                    value_copy[state] = q_value
            self.values = value_copy.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q_value = 0
        for trans_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            trans_reward = self.mdp.getReward(state, action, trans_state)
            state_reward = self.getValue(trans_state)
            q_value += prob * (trans_reward + self.discount * state_reward)
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        possible_actions = self.mdp.getPossibleActions(state)
        if len(possible_actions) == 0:
            return None
        counter = util.Counter()
        for action in possible_actions:
            counter[action] = self.getQValue(state, action)
        return counter.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        num_iterations = self.iterations
        states = self.mdp.getStates()
        for i in range(num_iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                q_value = self.getQValue(state, action)
                self.values[state] = q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # find all predecessors
        # set is used here to avoid redundancy
        predecessors = {state: set() for state in states}
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[next_state].add(state)

        pq = util.PriorityQueue()

        for state in states:
            if not self.mdp.isTerminal(state):
                max_q_value = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
                diff = abs(max_q_value - self.values[state])
                pq.push(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            pop_state = pq.pop()
            if not self.mdp.isTerminal(pop_state):
                max_q_value = max(self.getQValue(pop_state, action) for action in self.mdp.getPossibleActions(pop_state))
                self.values[pop_state] = max_q_value
                for predecessor in predecessors[pop_state]:
                    pred_max_q_value = max(self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor))
                    pred_diff = abs(pred_max_q_value - self.values[predecessor])
                    if pred_diff > self.theta:
                        pq.update(predecessor, -pred_diff)

