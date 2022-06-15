import random
import numpy as np
from graphviz import Digraph


# A transition class (transition of a an automaton). contains name, source state, target state
# and a reward (can be one of some possible values, depends on the automaton.
class Transition:
    def __init__(self, name, source_state, target_state, potential_reward=0):
        self.name = name
        self.source_state = source_state
        self.target_state = target_state
        self.potential_reward = potential_reward

    def get_reward(self, idx):
        return self.potential_reward[idx]

    def copy(self, status):
        return Transition(self.name, self.source_state, self.target_state, status)

    # a string representation of a transition - to be used in the execution report
    def __str__(self):
        return "{0} ---------{1}---------> {2}".format(self.source_state, self.name, self.target_state)

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return (self.source_state == other.source_state) \
               and (self.target_state == other.target_state) \
               and (self.name == other.name)


# The names is misleading - this is a finite and deterministic automaton (with only one accepting state)
# This automaton can be used as a "blackbox", if you use the "step" function.
class Process:
    def __init__(self, name, states=[], transitions=[], initial_state=None, accepting_states=None):
        self.name = name
        self.states = states  # list of names of the states.
        self.initial_state = initial_state
        self.accepting_states = accepting_states
        self.current_state = initial_state
        self.transitions = transitions  # transitions that are specific to the process.
        self.last_state = None

    def add_state(self, name):
        self.states.append(name)
        if self.current_state is None:
            self.initial_state = name
            self.current_state = name

    # return the current state of the process
    def get_current_state(self):
        return self.current_state

    # sets a new state as the current state (in case of triggering a transition for example)
    def set_current_state(self, name):
        if name in self.states:
            self.current_state = name

    def get_last_state(self):
        return self.last_state

    def set_last_state(self, name):
        if name in self.states or name is None:
            self.last_state = name

    # adds a new transition to the process
    def add_transition(self, name, source, target):
        self.transitions.append(Transition(name, source, target))

    # returns to the initial state of the transition
    def reset(self):
        self.set_current_state(self.initial_state)
        self.set_last_state(None)

    def is_accepting_state(self, state=None):
        if state is None:
            return self.current_state in self.accepting_states
        else:
            return state in self.accepting_states

    # returns the correct transition object according to its name and its source state
    # (the source state is enough in order to distinguish two transitions with the same name)
    def get_transition(self, tr_name, source_state=None):
        if source_state is None:
            source_state = self.current_state
        possible_tr = (tr for tr in self.transitions if tr.name == tr_name and tr.source_state == source_state)
        return next(possible_tr)

    # returns a set of names of transitions that can be triggered in the current state.
    def available_transitions(self):
        available = []
        for tr in self.transitions:
            if tr.source_state == self.current_state:
                available.append(tr.name)
        return available

    # switches the process' state according to the transition tr_name.
    def trigger_transition(self, tr_name):
        next_state = None

        for tr in self.transitions:
            if tr.name == tr_name and tr.source_state == self.current_state:
                next_state = tr.target_state
                break

        if next_state is not None:
            last_state = self.get_current_state()
            self.set_current_state(next_state)
            self.set_last_state(last_state)

    # chooses a random available state uniformly
    def get_random_transition(self):
        # return np.random.choice(self.available_transitions(), p=[0.6, 0.4])
        return random.choice(self.available_transitions())

    # checks if a certain transition is currently enabled
    def is_transition_enabled(self, tr_name):
        return tr_name in self.available_transitions()

    # this function chooses randomly the triggered transition, and then returns two values:
    # the transition which was chosen, and the reward that was given as a result.
    def step(self):
        rnd_tr = self.get_random_transition()
        full_rnd_tr = self.get_transition(rnd_tr)
        # trigger the chosen transition
        self.trigger_transition(rnd_tr)

        last_state = self.get_last_state()

        if self.is_accepting_state(last_state):
            reward = full_rnd_tr.get_reward(1)
        elif self.is_accepting_state():
            reward = full_rnd_tr.get_reward(0)
        else:
            reward = full_rnd_tr.get_reward(2)

        return rnd_tr, reward

    def visualize(self, directory=None):
        dfa_graph = Digraph()

        # vertices
        for st in self.states:
            if self.is_accepting_state(st):
                dfa_graph.attr('node', shape='doublecircle')
            else:
                dfa_graph.attr('node', shape='circle')
            dfa_graph.node(st)
        # edges
        for tr in self.transitions:
            dfa_graph.edge(tr.source_state, tr.target_state, tr.name)

        # init state arrow
        dfa_graph.attr('node', shape='none')
        dfa_graph.node('')
        dfa_graph.edge('', self.initial_state)

        if directory is not None:
            dfa_graph.render(self.name, view=False, directory=directory)
        else:
            dfa_graph.render(self.name, view=False)

    def is_prefix_accepted(self, prefix):
        self.reset()
        for c in prefix:
            self.trigger_transition(c)

        return self.is_accepting_state()

    def is_pos_neg_consistent(self, accepting_prefixes, rejecting_prefixes):
        for p in accepting_prefixes:
            if not self.is_prefix_accepted(p):
                return False

        for p in rejecting_prefixes:
            if self.is_prefix_accepted(p):
                return False

        return True
