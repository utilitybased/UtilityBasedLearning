import functools
from typing import Set
from automata.fa.dfa import DFA
import PTA
from PTA import State
from RLUtility import CommonUtils


class RPNI():
    def __init__(self, alphabet: Set[str], pos_examples: Set[str] = None, neg_examples: Set[str] = None):
        self._pos_examples = pos_examples
        self._neg_examples = neg_examples
        self._alphabet = alphabet
        if pos_examples is None and neg_examples is None:
            self._samples = None
        elif pos_examples is not None and neg_examples is None:
            self._samples = pos_examples
        elif pos_examples is None and neg_examples is not None:
            self._samples = neg_examples
        # both not None
        else:
            self._samples = pos_examples.union(neg_examples)

        self._red = {State('')}
        self._blue = set()

    def learn(self, initial_pta: PTA, blackbox: DFA) -> PTA:
        dfa = initial_pta

        pref_set = CommonUtils.prefix_set({s.name for s in initial_pta.pos_states})

        self._blue = {State(i) for i in self._alphabet.intersection(pref_set)}

        while len(self._blue) != 0:
            qb = pick_next_blue(self._blue)
            self._blue.remove(qb)

            found = False
            for qr in sorted(self._red, key=functools.cmp_to_key(_cmp)):

                if blackbox.accepts_input(qr.name) == blackbox.accepts_input(qb.name) and self.compatible_test(
                        self.try_merge(dfa.clone(), qr, qb), qr, qb):
                    dfa = self.try_merge(dfa, qr, qb)
                    new_blue_states = set()
                    for q in self._red:
                        for a in self._alphabet:
                            if dfa.is_transition_exists(q, a) and \
                                    dfa.get_transition(q, a) not in self._red:
                                new_blue_states.add(dfa.get_transition(q, a))

                    self._blue.update(new_blue_states)
                    found = True

            if not found:
                dfa = self.promote_state(qb, dfa)

        # positive only case
        if self._neg_examples is not None and self._pos_examples is None:
            for s in dfa.states:
                if not blackbox.accepts_input(s.name):
                    dfa.neg_states.add(s)

        return dfa

    def promote_state(self, qu: State, dfa: PTA) -> PTA:
        self._red.add(qu)

        self._blue.update({
            dfa.get_transition(qu, a) for a in self._alphabet if dfa.is_transition_exists(qu, a)
        })
        self._blue.discard(qu)

        return dfa

    def compatible_test(self, dfa: PTA, q1: State, q2: State) -> bool:

        return not any(dfa.get_output_4_prefix(w)[1] for w in self._neg_examples)

    def try_merge(self, dfa: PTA,
                  q: State,
                  q_prime: State) -> PTA:
        qf, a = dfa.find_transition_to_q(q_prime)

        if qf is None or a is None:
            return dfa

        dfa.add_transition(qf, q, a)
        fold_counter = 1
        dfa, fold_counter = self.fold(dfa, q, q_prime, fold_counter)

        # print("folds num: ", fold_counter)

        return dfa

    def fold(self, dfa: PTA,
             q: State,
             q_prime: State,
             fold_counter: int) -> PTA:

        if q_prime in dfa.pos_states:
            dfa.pos_states.add(q)

        for a in self._alphabet:
            if dfa.is_transition_exists(q_prime, a):
                if dfa.is_transition_exists(q, a):
                    dfa, fold_counter = self.fold(dfa, dfa.get_transition(q, a),
                                                  dfa.get_transition(q_prime, a), 1 + fold_counter)
                else:
                    dfa.add_transition(q, dfa.get_transition(q_prime, a), a)

        return dfa, fold_counter


def pick_next_blue(blue: Set[State]) -> State:
    return min(blue, key=functools.cmp_to_key(_cmp))

def _cmp(q1: State, q2: State) -> int:
    if len(q1.name) == len(q2.name):
        if q1.name > q2.name:
            return 1
        elif q1.name < q2.name:
            return -1
        else:
            return 0
    elif len(q1.name) > len(q2.name):
        return 1
    else:
        return -1
