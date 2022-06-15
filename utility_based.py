import itertools
import platform
from pathlib import Path

from automata.fa.dfa import DFA

from RLUtility import RLUtility, CommonUtils
from examples import *
from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
import time
from itertools import combinations

# remove unused states
def remove_unused_states(init_state, built_states, built_transitions):
    tr_to_remove = []
    st_to_remove = []
    for s in built_states:
        if s != init_state:
            unused = True
            for t in built_transitions:
                if t.target_state == s:
                    unused = False
            # if s is unused, remove it from built transitions
            if unused:
                st_to_remove.append(s)
                for t in built_transitions:
                    if t.source_state == s:
                        tr_to_remove.append(t)

    for t in tr_to_remove:
        built_transitions.remove(t)
    for s in st_to_remove:
        built_states.remove(s)


# using majority vote
def find_transition_target(action, prefixes, S2C, state_len):
    cluster_counter = [0]*len(set(S2C.values()))

    for p in prefixes:
        if len(p) < state_len and p+action in S2C:
            target_cluster = S2C[p+action]
            cluster_counter[int(target_cluster[1:])] += 1

    # print("{}, {}, {}, {}".format(S2C[prefixes[0]], action, prefixes, cluster_counter))
    max_val = max(cluster_counter)
    is_consistent = (max_val == sum(cluster_counter))

    return 'q' + str(cluster_counter.index(max_val))


# using shortest prefix
def find_transition_target2(action, prefixes, S2C, state_len):
    chosen_prefix = prefixes[0]
    return S2C[chosen_prefix+action]


def compatible_test_clustering(p1, p2, actions, S2C1, S2C2, state_len, acceptance_dict_tst):
    if S2C2[p1] == -1 or S2C2[p2] == -1:
        return True

    if S2C2[p1] != S2C2[p2]:
        if acceptance_dict_tst[p1] == acceptance_dict_tst[p2]:
            print(f"err for comparing({p1},{p2})")
        return False

    # both p are in kernel
    # use s2c1
    # if len(p1) < state_len and len(p2) < state_len:
    #     if S2C1[p1] != S2C1[p2]:
    #         if acceptance_dict_tst[p1] == acceptance_dict_tst[p2]:
    #             print(f"err for comparing({p1},{p2})")
    #         return False
    #
    # # use s2c2
    # else:
    #     if S2C2[p1] != S2C2[p2]:
    #         if acceptance_dict_tst[p1] == acceptance_dict_tst[p2]:
    #             print(f"err for comparing({p1},{p2})")
    #         return False

    # successors
    for act in actions:
        if not compatible_test_clustering(p1 + act, p2 + act, actions, S2C1, S2C2, state_len, acceptance_dict_tst):
            return False

    return True

def compatible_test_rec(p1, p2, actions, acceptance_dict):
    if acceptance_dict[p1] == -1 or acceptance_dict[p2] == -1:
        return True

    if acceptance_dict[p1] != acceptance_dict[p2]:
        return False

    for act in actions:
        if not compatible_test_rec(p1 + act, p2 + act, actions, acceptance_dict):
            return False

    return True

def read_input_state(dfa: DFA, input: str):
    start = dfa.initial_state
    for s in input:
        start = dfa.transitions[start][s]
    return start


def test_print(original_dfa, acceptance_dict, C2S):
    print("errors accepting")
    for k, v in acceptance_dict.items():
        if original_dfa.accepts_input(k) != acceptance_dict[k]:
            print(k, v)
    print("######")

    print("errors clustering")
    for cl1 in C2S.values():
        for cl2 in C2S.values():
            if cl1 != cl2:
                all_pairs = list(itertools.product(cl1, cl2))

                for p in all_pairs:
                    p1 = p[0]
                    p2 = p[1]

                    if read_input_state(original_dfa, p1) == read_input_state(original_dfa, p2):
                        print(p1, " and ", p2, " supposed to be in the same cluster")
    print("######")


def single_run(dfa: Process, params: GeneralParameters, Dual: bool = False):
    original_dfa = CommonUtils.process2dfa(dfa)

    Path(f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}").mkdir(parents=True, exist_ok=True)
    original_dfa.show_diagram(path=f"logs/{dfa.name}/{platform.node()}/original.png")
    file = open(f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}/prefix_tree_learning.dat", 'w+')

    actions = list(params.reward_dict.keys())

    start_monte_carlo = time.time()
    monte_carlo_p, samples_n = RLUtility.monte_carlo(dfa=dfa, params=params) #episode len is bigger
    end_monte_carlo = time.time()

    # clustering 1
    start_dbscan = time.time()

    values_to_process = np.array(list(monte_carlo_p.V1.values())).reshape(-1, 1)
    clustering_res = DBSCAN(eps=params.eps, min_samples=params.min_samples).fit(values_to_process)
    S2C = defaultdict(lambda: -1)
    for i in range(len(monte_carlo_p.states1)):
        S2C[monte_carlo_p.states1[i]] = "s" + str(clustering_res.labels_[i])
    C2S = RLUtility.invert_S2C(S2C)

    initial_clusters = list(set(S2C.values()))

    if Dual:
        # compatible clustering
        values_to_process = np.array(list(monte_carlo_p.V2.values())).reshape(-1, 1)
        clustering_res = DBSCAN(eps=params.eps2, min_samples=params.min_samples).fit(values_to_process)
        S2C2 = defaultdict(lambda: -1)
        for i in range(len(monte_carlo_p.states2)):
            S2C2[monte_carlo_p.states2[i]] = "s" + str(clustering_res.labels_[i])

        comp_clusters = list(set(S2C2.values()))

        print("comp clusters: ", comp_clusters)
        print("clusters after RL: ", initial_clusters)

    # After the policy evaluation step, we take the values of all prefixes an look for clusters using DBSCAN.
    end_dbscan = time.time()

    ############################################### End of RL Phase ###############################################

    # Set up acceptance data structures
    start_structure_build = time.time()

    acc_prefixes = list(monte_carlo_p.accepting_prefixes)
    rej_prefixes = list(monte_carlo_p.rejecting_prefixes)

    acceptance_dict = defaultdict(lambda: -1)
    for p in acc_prefixes:
        acceptance_dict[p] = True
    for p in rej_prefixes:
        acceptance_dict[p] = False

    end_structure_build = time.time()

    # logs
    # C2S2 = RLUtility.invert_S2C(S2C2)
    # for k,v in C2S2.items():
    #     res = list(combinations(v, 2))
    #     for p1, p2 in res:
    #         if acceptance_dict[p1] != acceptance_dict[p2]:
    #             print(f"err ({p1}-{monte_carlo_p.V2[p1]}, {p2}-{monte_carlo_p.V2[p2]}")

    # test
    # test_print(original_dfa, acceptance_dict, C2S)

    start_prefix_tree = time.time()
    new_clusters_count = 0
    for cluster in initial_clusters:
        buckets = []
        prefixes_in_cluster = C2S[cluster]

        if prefixes_in_cluster:
            buckets.append([])
            buckets[0].append(prefixes_in_cluster[0])
            for prefix_idx in range(1, len(prefixes_in_cluster)):
                bucket_idx = 0

                # compatible test for dual
                if Dual:
                    while bucket_idx < len(buckets) and (not compatible_test_clustering(p1=prefixes_in_cluster[prefix_idx],
                                                                                        p2=buckets[bucket_idx][0],
                                                                                        actions=actions,
                                                                                        S2C1=S2C,
                                                                                        S2C2=S2C2,
                                                                                        state_len=params.STATE_LEN,
                                                                                        acceptance_dict_tst=acceptance_dict)):
                        bucket_idx += 1
                else:
                    while bucket_idx < len(buckets) and (not compatible_test_rec(p1=prefixes_in_cluster[prefix_idx],
                                                                                    p2=buckets[bucket_idx][0],
                                                                                    actions=actions,
                                                                                    acceptance_dict=acceptance_dict)):
                        bucket_idx += 1

                if bucket_idx >= len(buckets):
                    buckets.append([])
                buckets[bucket_idx].append(prefixes_in_cluster[prefix_idx])

            # re-assign clusters to each prefix
            for bucket in buckets:
                for prefix in bucket:
                    S2C[prefix] = "q" + str(new_clusters_count)

                new_clusters_count += 1

    ######################################### End of Splitting Phase ############################################

    # clusters were re-assigned. Now we can construct the automaton.
    C2S = RLUtility.invert_S2C(S2C)

    # Inferring states
    built_states = list(set(S2C.values()))

    # Inferring the initial state.
    built_init_state = S2C['']

    # Inferring the accepting states
    built_accepting_states = set()
    for p in monte_carlo_p.states1:
        if p in monte_carlo_p.accepting_prefixes:
            built_accepting_states.add(S2C[p])

    # Inferring transitions
    built_transitions = []
    is_consistent = True
    for built_state in built_states:
        for action in actions:
            target_built_state = find_transition_target(action, C2S[built_state], S2C, params.STATE_LEN)
            built_transitions.append(Transition(action, built_state, target_built_state, 0))

    # remove unused states
    remove_unused_states(built_init_state, built_states, built_transitions)

    # Initializing the constructed dfa object.
    con_dfa = Process('constructed_dfa', states=built_states, transitions=built_transitions,
                      initial_state=built_init_state, accepting_states=built_accepting_states)
    end_prefix_tree = time.time()

    print("final clusters: ", built_states)

    ######################################### End of Construction Phase ############################################

    result_dfa = CommonUtils.process2dfa(con_dfa)
    result_dfa.show_diagram(path=f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}/{dfa.name}.png")

    time_elapsed = end_monte_carlo-start_monte_carlo + \
                   end_dbscan - start_dbscan + \
                   end_structure_build - start_structure_build + \
                   end_prefix_tree-start_prefix_tree

    print(f"{dfa.name} utility based learning - {time_elapsed} s")
    print(f"{dfa.name} utility based learning - {time_elapsed} s", file=file)

    # difference_dfa = result_dfa.difference(original_dfa, minify=True)
    # if not difference_dfa.isempty():
    #     difference_dfa.show_diagram(path=f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}/counterexample.png")

    return con_dfa, acc_prefixes, rej_prefixes, time_elapsed


if __name__ == '__main__':
    Dual = False
    RPNI = False
    dfa, params = example29(Dual=Dual, RPNI=RPNI)
    con_dfa, acc_prefixes, rej_prefixes, time_elapsed = single_run(dfa=dfa, params=params, Dual=Dual)

    # comparison between original and constructed dfa
    identical_dfa = con_dfa.is_pos_neg_consistent(acc_prefixes, rej_prefixes) and len(dfa.states) == len(con_dfa.states)
    print("Success: ", identical_dfa) # True means equal
