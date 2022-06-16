import functools
import random

import numpy as np
from collections import defaultdict
from tqdm import tqdm

from PTA import PTA
from PTA import State
import itertools
from get_strings import get_up_to_k_length
from process import Process
from automata.fa.dfa import DFA
from typing import Set, Generator, Tuple, Union, Any


class CommonUtils:
    @staticmethod
    def build_pta_from_blackbox(blackboxProc: Process,
                                blackbox: DFA,
                                num_iterations: int,
                                episode_len: int) -> Tuple[
        PTA, Set[Union[str, Any]], Set[Union[str, Any]]]:
        s_plus = set()
        s_minus = set()

        for it in tqdm(range(num_iterations)):
            episode = RLUtility.generate_episode(blackboxProc, episode_len)

            for step in episode:
                if blackbox.accepts_input(step[0]):
                    s_plus.add(step[0])
                else:
                    s_minus.add(step[0])

        samples = s_plus

        alphabet = blackbox.input_symbols#CommonUtils.determine_alphabet(samples)
        pta = PTA(alphabet)

        for letter in alphabet:
            pta.add_transition(State(''), State(letter), letter)

        # build pta from all sub prefixes
        states = {
            State(u) for u in CommonUtils.prefix_set(samples)
        }

        # add all transitions
        new_states = set()
        for u in states:
            for a in alphabet:
                ua = State(u.name + a)
                if ua not in states:
                    new_states.add(ua)

                pta.add_transition(u, ua, a)

        states.update(new_states)

        # set accept/reject
        for u in states:
            # if not bNegativeOnly and blackbox.accepts_input(u.name):
            if blackbox.accepts_input(u.name):
                pta.pos_states.add(u)

        pta.states = states

        return pta, s_plus, s_minus

    @staticmethod
    def prefix_set(s: Set[str]) -> Generator:
        for w in s:
            for i in range(len(w) + 1):
                yield w[:i]

    @staticmethod
    def process2dfa(proc: Process):
        states = {str(q) for q in proc.states}
        input_symbols = {t.name for t in proc.transitions}
        transitions = {}
        for s in states:
            n = {}
            for t in proc.transitions:
                if t.source_state == s:
                    n[t.name] = t.target_state
            transitions[s] = n
        final_states = {str(s) for s in proc.accepting_states if s in states}

        dfa = DFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state=proc.initial_state,
            final_states=final_states
        )

        return dfa

    @staticmethod
    def determine_alphabet(s: Set[str]) -> Set[str]:
        return set(''.join(s))

    @staticmethod
    def pta2dfa(pta: PTA):
        states = {str(q.name) for q in pta.states}
        input_symbols = pta.alphabet

        _transitions = {}
        for source, dict_target in pta._transitions.items():
            _transitions[source.name] = dict()
            for a, s_target in dict_target.items():
                _transitions[source.name][a] = s_target.name

        final_states = {s.name for s in pta.pos_states}
        dfa = DFA(
            states=states,
            input_symbols=input_symbols,
            transitions=_transitions,
            initial_state=pta._initial_state.name,
            final_states=final_states
        )
        return dfa

class MonteCarloParams:
    def __init__(self,
                 V1,
                 V2,
                 states1,
                 states2,
                 accepting_prefixes,
                 rejecting_prefixes):
        self.V1 = V1
        self.V2 = V2
        self.states1 = states1
        self.states2 = states2
        self.accepting_prefixes = accepting_prefixes
        self.rejecting_prefixes = rejecting_prefixes

class RLUtility(object):
    @staticmethod
    # S2C - maps prefixes to different clusters
    # C2S - the other way around
    def invert_S2C(dic):
        inv = {}
        for key, val in dic.items():
            inv[val] = inv.get(val, []) + [key]
        return inv

    @staticmethod
    # this function generates a single random episode from the blackbox, and returns it. In this context, "init_state" and
    # "final_state" are prefixes.
    def generate_episode(dfa, episode_length):
        init_state = ""
        episode = []

        dfa.reset()
        for _ in range(episode_length):
            action, reward = dfa.step()
            final_state = init_state + action
            episode.append([init_state, action, reward, final_state])
            init_state = final_state

        return episode

    @staticmethod
    def monte_carlo(dfa, params):
        if not params.COMP_LAYER_LEN:
            params.COMP_LAYER_LEN = 0

        # algorithm initialization
        states = get_up_to_k_length(params.actions, params.STATE_LEN + params.COMP_LAYER_LEN)
        states_size = len(states)
        accepting_prefixes = set()
        rejecting_prefixes = set()

        V = {s: 0 for s in states}
        returns_sum = defaultdict(lambda: 0)
        returns_count = defaultdict(lambda: 0)

        for it in tqdm(range(params.num_iterations)):
            episode = RLUtility.generate_episode(dfa, params.episode_len)
            G = 0
            # print(episode)

            if len(episode) > 0:
                for i, step in enumerate(episode[::-1]):
                    G = params.gamma * G + step[2]
                    if i > len(episode) - params.STATE_LEN - 2 - params.COMP_LAYER_LEN:
                        returns_sum[step[0]] += G
                        returns_count[step[0]] += 1
                        V[step[0]] = returns_sum[step[0]] / returns_count[step[0]]

                    if step[2] in params.accepting_states_rewards:
                        accepting_prefixes.add(step[0])
                    else:
                        rejecting_prefixes.add(step[0])
        #remove keys:
        for k in states:
            if k not in accepting_prefixes and k not in rejecting_prefixes:
                V.pop(k, None)

        # V1 is the "good" clustering while V2 use for compatible only
        V1 = {pre: val for (pre, val) in V.items() if len(pre) <= params.STATE_LEN}
        V2 = V

        print("Number of accepting strings: {}".format(len(accepting_prefixes)))
        print("Number of rejecting strings: {}".format(len(rejecting_prefixes)))
        print("Total number of strings: {}".format(len(accepting_prefixes) + len(rejecting_prefixes)))
        # print(V)

        # After the policy evaluation step, we take the values of all prefixes an look for clusters using DBSCAN.
        p = MonteCarloParams(V1=V1, V2=V2, states1=list(V1.keys()), states2=list(V2.keys()), rejecting_prefixes=rejecting_prefixes,
                             accepting_prefixes=accepting_prefixes)

        return p, len(accepting_prefixes) + len(rejecting_prefixes)
