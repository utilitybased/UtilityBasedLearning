import graphviz
import copy
from collections import defaultdict, OrderedDict, deque
from typing import Set, Tuple, List, Generator
import abc


class State:

    def __init__(self, name: str):
        self.__name = name

    @property
    def name(self):
        return self.__name

    def __hash__(self):
        return hash(self.__name)

    def __eq__(self, other):
        return self.__name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name

    def __le__(self, other):
        return self.name <= other.name

    def __ge__(self, other):
        return self.name >= other.name

    def __ne__(self, other):
        return self.name != other.name

    def __str__(self):
        return self.name


class BaseTree(abc.ABC):
    def __init__(self, alphabet: Set[str]):
        if '' in alphabet:
            raise ValueError('The empty string is not allowed in the alphabet!')
        self.__alphabet = alphabet

    @property
    def alphabet(self):
        return self.__alphabet

    @abc.abstractmethod
    def get_output_4_prefix(self, s: str) -> Tuple[State, bool]:
        pass


class PTA(BaseTree):
    def __init__(self, alphabet: Set[str], initial_state: State = State('')):
        super().__init__(alphabet)

        self._initial_state = initial_state

        self.states = {self._initial_state}

        self.pos_states = set()

        self.neg_states = set()

        self._transitions = defaultdict(OrderedDict)

    def get_output_4_prefix(self, s: str) -> Tuple[State, bool]:
        q = self._initial_state
        for letter in s:
            if not self.is_transition_exists(q, letter):
                return q, False
            q = self.get_transition(q, letter)

        return q, q in self.pos_states

    def add_transition(self, q1: State, q2: State, a: str):
        if a not in self.alphabet:
            raise ValueError('\'{}\' is not in the alphabet of the dfa!'.format(a))

        self.states.update({q1, q2})
        self._transitions[q1][a] = q2

    def is_transition_exists(self, q1: State, a: str) -> bool:
        return q1 in self._transitions and \
               a in self._transitions[q1] and \
               self._transitions[q1][a] in self.states

    def get_transition(self, q1: State, a: str) -> State:
        return self._transitions[q1][a]

    def find_transition_to_q(self, q: State) -> Tuple:
        for qf in self._transitions.keys():
            for letter, to_state in self._transitions[qf].items():
                if to_state == q:
                    return qf, letter
        return None, None

    def find_transitions_to_q_with_letter(self, q: State, a: str) -> Set[State]:
        states = set()
        for qf in self._transitions.keys():
            for letter, to_state in self._transitions[qf].items():
                if to_state == q and letter == a:
                    states.add(qf)
        return states

    def force_minimize(self):
        p = self._hopcroft()

        start = [state_set for state_set in p if self._initial_state in state_set]
        assert len(start) == 1

        minimized_dfa = PTA(self.alphabet, State(''.join(map(str, start[0]))))
        for state_set in p:
            for a in self.alphabet:
                for state in state_set:
                    if self.is_transition_exists(state, a):
                        to_state = self.get_transition(state, a)
                        to = [s for s in p if to_state in s]
                        assert len(to) == 1

                        to_state_set = to[0]
                        minimized_dfa.add_transition(State(''.join(map(str, state_set))),
                                                     State(''.join(map(str, to_state_set))),
                                                     a)
                        break

            if any(s in self.pos_states for s in state_set):
                minimized_dfa.pos_states.add(State(''.join(map(str, state_set))))

        return minimized_dfa.rename_states()

    def _hopcroft(self):
        qf = self.states - self.pos_states
        if len(self.pos_states) < len(self.states - self.pos_states):
            p = [qf, self.pos_states]
            l = deque([self.pos_states])
        else:
            p = [self.pos_states, qf]
            l = deque([qf])

        while len(l) > 0:
            s = l.popleft()
            for a in self.alphabet:
                for b in p.copy():
                    b1, b2 = self._split(b, s, a)
                    p.remove(b)
                    if len(b1) > 0:
                        p.append(b1)
                    if len(b2) > 0:
                        p.append(b2)

                    if len(b1) < len(b2):
                        if len(b1) > 0:
                            l.append(b1)
                    else:
                        if len(b2) > 0:
                            l.append(b2)
        return p

    def _split(self, b_prime, b, a):
        ba = set()
        for state in b:
            ba.update(self.find_transitions_to_q_with_letter(state, a))

        ba_comp = ba.symmetric_difference(self.states)
        return b_prime.intersection(ba), b_prime.intersection(ba_comp)

    def rename_states(self):
        alphabet = sorted(self.alphabet)
        dfa = PTA(self.alphabet, State('s0'))

        queue = deque([self._initial_state])
        visited = {self._initial_state}
        cnt = 1
        old_to_new = {self._initial_state: State('s0')}

        while len(queue) > 0:
            state = queue.popleft()

            for a in alphabet:
                if state in self._transitions and a in self._transitions[state]:
                    to_state = self.get_transition(state, a)

                    if to_state not in visited:
                        queue.append(to_state)
                        visited.add(to_state)

                        old_to_new[to_state] = State('s' + str(cnt))
                        cnt += 1

        for old, new in old_to_new.items():
            old_transitions = self._transitions[old]

            for sym, state in old_transitions.items():
                dfa.add_transition(new, old_to_new[state], sym)

            if old in self.pos_states:
                dfa.pos_states.add(new)
            elif old in self.neg_states:
                dfa.neg_states.add(new)
        return dfa

    def clone(self):
        cp = PTA(self.alphabet, initial_state=self._initial_state)

        cp.states = self.states.copy()
        cp.pos_states = self.pos_states.copy()
        cp.neg_states = self.neg_states.copy()
        cp._transitions = copy.deepcopy(self._transitions)

        return cp

    def view_object(self) -> graphviz.Digraph:
        digraph = graphviz.Digraph('dfa')
        digraph.graph_attr['rankdir'] = 'LR'

        edges = defaultdict(lambda: defaultdict(list))

        for state in self.states:
            shape = 'doublecircle' if state in self.pos_states else 'circle'
            digraph.node(name='q{}'.format(state.name), shape=shape, constraint='false')

            if state in self._transitions:
                for letter, to_state in self._transitions[state].items():
                    edges[state][to_state].append(letter)

        for from_state in edges:
            for to_state, letters in edges[from_state].items():
                digraph.edge('q{}'.format(from_state.name),
                             'q{}'.format(to_state.name),
                             ', '.join(letters))

        digraph.node('', shape='plaintext', constraint='true')
        digraph.edge('', 'q{}'.format(self._initial_state.name))

        return digraph

    def visualize(self):
        self.view_object().view()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        tests = self._initial_state == other._initial_state and \
                self.states == other.states and \
                self.pos_states == other.pos_states

        if not tests:
            return False

        if sorted(self._transitions.keys()) != sorted(other._transitions.keys()):
            return False

        for k in self._transitions.keys():
            if sorted(self._transitions[k].keys()) != sorted(other._transitions[k].keys()):
                return False

            if sorted(self._transitions[k].values()) != sorted(other._transitions[k].values()):
                return False
        return True

    def __str__(self):
        rep = [
            'Initial state:    = {}'.format(self._initial_state),
            'Alphabet:         = {}'.format(self.alphabet),
            'States:           = {}'.format(set(map(str, self.states))),
            'Accepting states: = {}'.format(set(map(str, self.pos_states))),
            'Rejecting states: = {}'.format(set(map(str, self.neg_states))),
            '\nTransition function: delta'
        ]

        for state in sorted(self._transitions.keys()):
            rep.append('state = q_{}'.format(state))
            for letter, to_state in self._transitions[state].items():
                rep.append('delta(q_{}, {}) = q_{}'.format(state, letter, to_state))
            rep.append('')

        return '\n'.join(rep)
