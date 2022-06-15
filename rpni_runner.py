import random
import sys
from itertools import combinations

from termcolor import colored
from automata.fa.dfa import DFA
import platform
from pathlib import Path
from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
from PTA import PTA
from RLUtility import RLUtility, CommonUtils
from RPNI import RPNI
from examples import *
import time


def random_validator_pta(result_dfa: PTA, original_dfa: DFA):
    for _ in range(1000):
        random_str = ''.join(random.choice(list(original_dfa.input_symbols)) for i in
                             range(random.randint(0, len(original_dfa.states) * 5)))

        if original_dfa.accepts_input(random_str) != result_dfa.get_output_4_prefix(random_str)[1]:
            return False

    return True


def random_validator(result_dfa: DFA, original_dfa: DFA):
    if len(result_dfa.states) != len(original_dfa.states):
        return False

    for _ in range(1000):
        random_str = ''.join(random.choice(list(original_dfa.input_symbols)) for i in
                             range(random.randint(0, len(original_dfa.states) * 5)))

        if original_dfa.accepts_input(random_str) != result_dfa.accepts_input(random_str):
            return False

    return True


def learn_4_parameters_prefix(original_process: Process, original_dfa: DFA, num_iterations_param: int, episode_len_param: int):
    start_time = time.time()

    # build pta
    pta, pos_examples, neg_examples = CommonUtils.build_pta_from_blackbox(blackboxProc=original_process,
                                                                          blackbox=original_dfa,
                                                                          episode_len=episode_len_param,
                                                                          num_iterations=num_iterations_param)


    learner = RPNI(neg_examples=neg_examples, alphabet=original_dfa.input_symbols)

    result = learner.learn(blackbox=original_dfa, initial_pta=pta)

    total_samples = len(neg_examples) + len(pta.pos_states)

    print(f"total examples: {total_samples} (pos-{len(pta.pos_states)}, neg-{len(neg_examples)})")

    # learn traditional requires both pos and neg for learner
    # learner = RPNI(pos_examples=pos_examples, neg_examples=neg_examples, alphabet=original_dfa.input_symbols)
    # result = learner.learn_traditional(initial_pta=pta)
    # total_samples = len(neg_examples) + len(pos_examples)
    # print(f"total examples: {total_samples} (pos-{len(pos_examples)}, neg-{len(neg_examples)})")

    end_time = time.time()
    # print("&&", end_time-start_time)
    try:
        # result.show()
        result_dfa = CommonUtils.pta2dfa(result)
    except Exception as e:
        # result.show()
        print(e)
        if random_validator_pta(result, original_dfa):
            print("Equal?", True, f"total_samples: {total_samples} time: {end_time - start_time}")
            return original_dfa, total_samples, end_time - start_time
        else:
            print("Equal?", False)
            return None, None, None

    # validate
    d1 = result_dfa.difference(original_dfa)
    d2 = original_dfa.difference(result_dfa)

    # random validator
    isEq = d1.isempty() and d2.isempty() and random_validator(result_dfa, original_dfa)
    print("Equal?", isEq, f"total_samples: {total_samples} time: {end_time - start_time}" if isEq else '')

    if isEq:
        # print("removed states:", len(pta.states) - len(result.states))
        return result_dfa, total_samples, end_time - start_time
    else:
        return None, None, None


def find_optimum_run(episode_len_param: int, num_iterations_param: int, validate_iterations: int = 10):
    is_find_optimal_params = False
    is_worked = False

    min_time = sys.maxsize
    min_samples = sys.maxsize
    result_dfa_print = None
    num_iterations_optimum = sys.maxsize
    episode_len_optimum = sys.maxsize

    while not is_find_optimal_params:
        # check validation 10 times
        n_faults = 0
        avg_time = 0
        avg_samples = 0
        result_dfa_working = None
        for i in range(validate_iterations):
            result_dfa, total_samples, total_time = learn_4_parameters_prefix(original_process=dfa,
                                                                              original_dfa=original_dfa,
                                                                              num_iterations_param=num_iterations_param,
                                                                              episode_len_param=episode_len_param)

            if result_dfa is None and n_faults <= 0.2 * validate_iterations:
                is_find_optimal_params = True
                break
            else:
                if i == validate_iterations - 1:
                    result_dfa_working = result_dfa
                is_worked = True
                avg_time += total_time
                avg_samples += total_samples

        # optimum numbers are the numbers worked after 10 iterations
        # result_dfa_working is not None after 10 iterations
        if is_worked and result_dfa_working is not None:
            result_dfa_print = result_dfa_working
            min_time = avg_time / validate_iterations
            min_samples = avg_samples / validate_iterations
            num_iterations_optimum = num_iterations_param
            episode_len_optimum = episode_len_param

        num_iterations_param = num_iterations_param - 5
        # I'm not sure if I want to change this here
        # episode_len = episode_len - 1

    if result_dfa_print is None:
        print("learning failed for initial parameters")
        return False
    else:

        str1 = f"RPNI time building: {min_time}s \n samples number: {round(min_samples)}"

        Path(f"logs/{dfa.name}/{platform.node()}/{num_iterations_optimum}").mkdir(parents=True, exist_ok=True)
        file = open(f"logs/{dfa.name}/{platform.node()}/{num_iterations_optimum}/rpni_learning.dat", 'w+')

        print(colored(str1, 'green'))

        print(str1, file=file)

        name_png = f"{dfa.name}_rpni.png"

        try:
            result_dfa_print.show_diagram(
                path=f"logs/{dfa.name}/{platform.node()}/{num_iterations_optimum}/{name_png}")
        except Exception as e:
            print("problem in show generated diagram")
            print(e)

        return True


if __name__ == '__main__':
    dfa, params = example29(RPNI=True, Dual=False)
    validate_iter = 10
    episode_len = params.episode_len
    num_iterations = params.num_iterations

    original_dfa = CommonUtils.process2dfa(dfa)

    Path(f"logs/{dfa.name}/{platform.node()}").mkdir(parents=True, exist_ok=True)
    try:
        original_dfa.show_diagram(path=f"logs/{dfa.name}/{platform.node()}/original.png")
    except Exception as e:
        print("problem show diagram")
        print(e)


    print("#######################")
    found = False
    stop_loop = time.time()
    itr = 0
    while not found:
        if time.time() - stop_loop > 100:
            break
        found = find_optimum_run(episode_len_param=episode_len,# + itr * 1,
                                 num_iterations_param=num_iterations + itr * 5,
                                 validate_iterations=validate_iter)

        itr += 1

