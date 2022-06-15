from process import Transition, Process


class GeneralParameters:
    def __init__(self,
                 STATE_LEN=8,
                 COMP_LAYER_LEN=2,
                 gamma=0.5,
                 reward_dict={'a': [20, 100, 0], 'b': [40, 60, 0]},
                 episode_len=15,
                 num_iterations=150,
                 eps=26,
                 eps2=26,
                 min_samples=1):
        self.STATE_LEN = STATE_LEN
        self.COMP_LAYER_LEN = COMP_LAYER_LEN
        self.gamma = gamma
        self.reward_dict = reward_dict
        self.episode_len = episode_len
        self.num_iterations = num_iterations
        self.eps = eps
        self.eps2 = eps2
        self.min_samples = min_samples
        self.actions = list(self.reward_dict.keys())
        self.accepting_states_rewards = [self.reward_dict[a][1] for a in self.actions]

    def sprint_details(self):
        return f"STATE_LEN: {self.STATE_LEN} " \
               f"gamma: {self.gamma} " \
               f"reward_dict: {self.reward_dict} " \
               f"episode_len: {self.episode_len} " \
               f"num_iterations: {self.num_iterations} " \
               f"eps: {self.eps} " \
               f"min_samples: {self.min_samples} " \
               f"actions: {self.actions} " \
               f"accepting_states_rewards: {self.accepting_states_rewards} "


# blackbox DFA initialization - the special counterexample
# single cross
def example8(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        COMP_LAYER_LEN = 3
        STATE_LEN = 6
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        episode_len = 11
        num_iterations = 180
        eps = 120
        eps2 = 200

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        # general parameters
        STATE_LEN = 6
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        episode_len = 9
        num_iterations = 230
        eps = 120

        if RPNI:
            episode_len = 8
            num_iterations = 45

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's1', 's2', reward_dict['a'])
    t2 = Transition('b', 's1', 's3', reward_dict['b'])
    t3 = Transition('a', 's2', 's5', reward_dict['a'])
    t4 = Transition('b', 's2', 's4', reward_dict['b'])
    t5 = Transition('a', 's3', 's4', reward_dict['a'])
    t6 = Transition('b', 's3', 's5', reward_dict['b'])
    t7 = Transition('a', 's4', 's6', reward_dict['a'])
    t8 = Transition('b', 's4', 's7', reward_dict['b'])
    t9 = Transition('a', 's5', 's7', reward_dict['a'])
    t10 = Transition('b', 's5', 's6', reward_dict['b'])
    t11 = Transition('a', 's6', 's7', reward_dict['a'])
    t12 = Transition('b', 's6', 's7', reward_dict['b'])
    t13 = Transition('a', 's7', 's7', reward_dict['a'])
    t14 = Transition('b', 's7', 's7', reward_dict['b'])
    dfa = Process('single_cross', states=['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14],
                  initial_state='s1', accepting_states=['s6'])

    return dfa, params


# combination lock (7)
def example9(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 8  # longest prefix in memory
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 15
        num_iterations = 20000
        eps = 24
        eps2 = 75
        COMP_LAYER_LEN = 4

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        # general parameters
        STATE_LEN = 8  # longest prefix in memory
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 13
        num_iterations = 19000
        eps = 26

        if RPNI:
            episode_len = 8
            num_iterations = 190

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's0', reward_dict['b'])
    t3 = Transition('a', 's1', 's2', reward_dict['a'])
    t4 = Transition('b', 's1', 's0', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's0', reward_dict['b'])
    t7 = Transition('a', 's3', 's0', reward_dict['a'])
    t8 = Transition('b', 's3', 's4', reward_dict['b'])
    t9 = Transition('a', 's4', 's0', reward_dict['a'])
    t10 = Transition('b', 's4', 's5', reward_dict['b'])
    t11 = Transition('a', 's5', 's6', reward_dict['a'])
    t12 = Transition('b', 's5', 's0', reward_dict['b'])
    t13 = Transition('a', 's6', 's6', reward_dict['a'])
    t14 = Transition('b', 's6', 's6', reward_dict['b'])
    dfa = Process('combination_lock7', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14],
                  initial_state='s0', accepting_states=['s6'])

    return dfa, params


# double cross
def example11(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 6  # longest prefix in memory
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 11
        num_iterations = 180
        eps = 70
        eps2 = 200
        COMP_LAYER_LEN = 3

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        # general parameters
        STATE_LEN = 6
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 10
        num_iterations = 300
        eps = 70  # 70

        if RPNI:
            episode_len = 10
            num_iterations = 50

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's2', reward_dict['b'])
    t3 = Transition('a', 's1', 's3', reward_dict['a'])
    t4 = Transition('b', 's1', 's4', reward_dict['b'])
    t5 = Transition('a', 's2', 's5', reward_dict['a'])
    t6 = Transition('b', 's2', 's6', reward_dict['b'])
    t7 = Transition('a', 's3', 's7', reward_dict['a'])
    t8 = Transition('b', 's3', 's9', reward_dict['b'])
    t9 = Transition('a', 's4', 's10', reward_dict['a'])
    t10 = Transition('b', 's4', 's8', reward_dict['b'])
    t11 = Transition('a', 's5', 's9', reward_dict['a'])
    t12 = Transition('b', 's5', 's7', reward_dict['b'])
    t13 = Transition('a', 's6', 's8', reward_dict['a'])
    t14 = Transition('b', 's6', 's10', reward_dict['b'])
    t15 = Transition('a', 's7', 's11', reward_dict['a'])
    t16 = Transition('b', 's7', 's12', reward_dict['b'])
    t17 = Transition('a', 's8', 's11', reward_dict['a'])
    t18 = Transition('b', 's8', 's12', reward_dict['b'])
    t19 = Transition('a', 's9', 's12', reward_dict['a'])
    t20 = Transition('b', 's9', 's11', reward_dict['b'])
    t21 = Transition('a', 's10', 's12', reward_dict['a'])
    t22 = Transition('b', 's10', 's11', reward_dict['b'])
    t23 = Transition('a', 's11', 's12', reward_dict['a'])
    t24 = Transition('b', 's11', 's12', reward_dict['b'])
    t25 = Transition('a', 's12', 's12', reward_dict['a'])
    t26 = Transition('b', 's12', 's12', reward_dict['b'])
    dfa = Process('double_cross', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
                                          's8', 's9', 's10', 's11', 's12'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16,
                               t17, t18, t19, t20, t21, t22, t23, t24, t25, t26],
                  initial_state='s0', accepting_states=['s11'])

    return dfa, params


# triple cross
def example12(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 7  # longest prefix in memory
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 15
        num_iterations = 1300
        eps = 45
        eps2 = 100
        COMP_LAYER_LEN = 4

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        # general parameters
        STATE_LEN = 7
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 11
        num_iterations = 1800
        eps = 45  # 70

        if RPNI:
            episode_len = 12
            num_iterations = 50

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's2', reward_dict['b'])
    t3 = Transition('a', 's1', 's4', reward_dict['a'])
    t4 = Transition('b', 's1', 's3', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's4', reward_dict['b'])
    t7 = Transition('a', 's3', 's6', reward_dict['a'])
    t8 = Transition('b', 's3', 's5', reward_dict['b'])
    t9 = Transition('a', 's4', 's5', reward_dict['a'])
    t10 = Transition('b', 's4', 's6', reward_dict['b'])
    t11 = Transition('a', 's5', 's8', reward_dict['a'])
    t12 = Transition('b', 's5', 's7', reward_dict['b'])
    t13 = Transition('a', 's6', 's7', reward_dict['a'])
    t14 = Transition('b', 's6', 's8', reward_dict['b'])
    t15 = Transition('a', 's7', 's9', reward_dict['a'])
    t16 = Transition('b', 's7', 's10', reward_dict['b'])
    t17 = Transition('a', 's8', 's10', reward_dict['a'])
    t18 = Transition('b', 's8', 's9', reward_dict['b'])
    t19 = Transition('a', 's9', 's10', reward_dict['a'])
    t20 = Transition('b', 's9', 's10', reward_dict['b'])
    t21 = Transition('a', 's10', 's10', reward_dict['a'])
    t22 = Transition('b', 's10', 's10', reward_dict['b'])
    dfa = Process('triple_cross', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
                                          's8', 's9', 's10'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16,
                               t17, t18, t19, t20, t21, t22],
                  initial_state='s0', accepting_states=['s9'])

    return dfa, params


# quadruple cross
def example13(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 8  # longest prefix in memory
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 16
        num_iterations = 2100
        eps = 30
        eps2 = 80
        COMP_LAYER_LEN = 4

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        # general parameters
        STATE_LEN = 8
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 12
        num_iterations = 3000
        eps = 30  # 70

        if RPNI:
            episode_len = 14
            num_iterations = 45

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's2', reward_dict['b'])
    t3 = Transition('a', 's1', 's4', reward_dict['a'])
    t4 = Transition('b', 's1', 's3', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's4', reward_dict['b'])
    t7 = Transition('a', 's3', 's6', reward_dict['a'])
    t8 = Transition('b', 's3', 's5', reward_dict['b'])
    t9 = Transition('a', 's4', 's5', reward_dict['a'])
    t10 = Transition('b', 's4', 's6', reward_dict['b'])
    t11 = Transition('a', 's5', 's8', reward_dict['a'])
    t12 = Transition('b', 's5', 's7', reward_dict['b'])
    t13 = Transition('a', 's6', 's7', reward_dict['a'])
    t14 = Transition('b', 's6', 's8', reward_dict['b'])
    t15 = Transition('a', 's7', 's10', reward_dict['a'])
    t16 = Transition('b', 's7', 's9', reward_dict['b'])
    t17 = Transition('a', 's8', 's9', reward_dict['a'])
    t18 = Transition('b', 's8', 's10', reward_dict['b'])

    t19 = Transition('a', 's9', 's11', reward_dict['a'])
    t20 = Transition('b', 's9', 's12', reward_dict['b'])
    t21 = Transition('a', 's10', 's12', reward_dict['a'])
    t22 = Transition('b', 's10', 's11', reward_dict['b'])

    t23 = Transition('a', 's11', 's12', reward_dict['a'])
    t24 = Transition('b', 's11', 's12', reward_dict['b'])
    t25 = Transition('a', 's12', 's12', reward_dict['a'])
    t26 = Transition('b', 's12', 's12', reward_dict['b'])
    dfa = Process('quadruple_cross', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
                                             's8', 's9', 's10', 's11', 's12'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16,
                               t17, t18, t19, t20, t21, t22, t23, t24, t25, t26],
                  initial_state='s0', accepting_states=['s11'])

    return dfa, params


# quintuple cross
def example14(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 8  # longest prefix in memory
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 16
        num_iterations = 5000
        eps = 18
        eps2 = 70
        COMP_LAYER_LEN = 4

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        # general parameters
        STATE_LEN = 8
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 1200, 0], 'b': [400, 800, 0]}
        episode_len = 13
        num_iterations = 2000
        eps = 18  # 70

        if RPNI:
            episode_len = 16
            num_iterations = 60

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's2', reward_dict['b'])
    t3 = Transition('a', 's1', 's4', reward_dict['a'])
    t4 = Transition('b', 's1', 's3', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's4', reward_dict['b'])
    t7 = Transition('a', 's3', 's6', reward_dict['a'])
    t8 = Transition('b', 's3', 's5', reward_dict['b'])
    t9 = Transition('a', 's4', 's5', reward_dict['a'])
    t10 = Transition('b', 's4', 's6', reward_dict['b'])
    t11 = Transition('a', 's5', 's8', reward_dict['a'])
    t12 = Transition('b', 's5', 's7', reward_dict['b'])
    t13 = Transition('a', 's6', 's7', reward_dict['a'])
    t14 = Transition('b', 's6', 's8', reward_dict['b'])
    t15 = Transition('a', 's7', 's10', reward_dict['a'])
    t16 = Transition('b', 's7', 's9', reward_dict['b'])
    t17 = Transition('a', 's8', 's9', reward_dict['a'])
    t18 = Transition('b', 's8', 's10', reward_dict['b'])
    t19 = Transition('a', 's9', 's12', reward_dict['a'])
    t20 = Transition('b', 's9', 's11', reward_dict['b'])
    t21 = Transition('a', 's10', 's11', reward_dict['a'])
    t22 = Transition('b', 's10', 's12', reward_dict['b'])
    t23 = Transition('a', 's11', 's13', reward_dict['a'])
    t24 = Transition('b', 's11', 's14', reward_dict['b'])
    t25 = Transition('a', 's12', 's14', reward_dict['a'])
    t26 = Transition('b', 's12', 's13', reward_dict['b'])
    t27 = Transition('a', 's13', 's14', reward_dict['a'])
    t28 = Transition('b', 's13', 's14', reward_dict['b'])
    t29 = Transition('a', 's14', 's14', reward_dict['a'])
    t30 = Transition('b', 's14', 's14', reward_dict['b'])
    dfa = Process('quintuple_cross', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
                                             's8', 's9', 's10', 's11', 's12', 's13', 's14'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16,
                               t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30],
                  initial_state='s0', accepting_states=['s13'])

    return dfa, params


# combination lock (6)
def example15(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 7  # longest prefix in memory
        COMP_LAYER_LEN = 3
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 14
        num_iterations = 20000
        # eps = 10#70
        # eps2 = 30
        eps = 30
        eps2 = 200
        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        STATE_LEN = 7  # longest prefix in memory
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 11
        num_iterations = 4000
        eps = 70

        if RPNI:
            episode_len = 6
            num_iterations = 180

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)


    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's0', reward_dict['b'])
    t3 = Transition('a', 's1', 's2', reward_dict['a'])
    t4 = Transition('b', 's1', 's0', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's0', reward_dict['b'])
    t7 = Transition('a', 's3', 's0', reward_dict['a'])
    t8 = Transition('b', 's3', 's4', reward_dict['b'])
    t9 = Transition('a', 's4', 's0', reward_dict['a'])
    t10 = Transition('b', 's4', 's5', reward_dict['b'])
    t11 = Transition('a', 's5', 's5', reward_dict['a'])
    t12 = Transition('b', 's5', 's5', reward_dict['b'])
    dfa = Process('combination_lock6', states=['s0', 's1', 's2', 's3', 's4', 's5'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12],
                  initial_state='s0', accepting_states=['s5'])

    return dfa, params


# combination lock (5)
def example16(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 6  # longest prefix in memory
        COMP_LAYER_LEN = 3
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 12
        num_iterations = 7100
        # eps = 10#70
        # eps2 = 30
        eps = 20
        eps2 = 50
        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        STATE_LEN = 6  # longest prefix in memory
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 10
        num_iterations = 250
        eps = 50

        if RPNI:
            episode_len = 5
            num_iterations = 150

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    if params is None:
        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's0', reward_dict['b'])
    t3 = Transition('a', 's1', 's2', reward_dict['a'])
    t4 = Transition('b', 's1', 's0', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's0', reward_dict['b'])
    t7 = Transition('a', 's3', 's0', reward_dict['a'])
    t8 = Transition('b', 's3', 's4', reward_dict['b'])
    t9 = Transition('a', 's4', 's4', reward_dict['a'])
    t10 = Transition('b', 's4', 's4', reward_dict['b'])
    dfa = Process('combination_lock5', states=['s0', 's1', 's2', 's3', 's4'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10],
                  initial_state='s0', accepting_states=['s4'])

    return dfa, params


# combination lock (4)
def example17(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 5  # longest prefix in memory
        COMP_LAYER_LEN = 2
        gamma = 0.6  # discount factor
        # gamma = 0.3  # discount factor
        # reward_dict = {'a': [10, 40, 0], 'b': [5, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 10
        num_iterations = 400
        # eps = 10#70
        # eps2 = 30
        eps = 30
        eps2 = 60
        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2,
                                   COMP_LAYER_LEN=COMP_LAYER_LEN)
    else:
        STATE_LEN = 5  # longest prefix in memory
        gamma = 0.6  # discount factor
        reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 8
        num_iterations = 100
        eps = 60

        if RPNI:
            episode_len = 5
            num_iterations = 50

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('a', 's0', 's1', reward_dict['a'])
    t2 = Transition('b', 's0', 's0', reward_dict['b'])
    t3 = Transition('a', 's1', 's2', reward_dict['a'])
    t4 = Transition('b', 's1', 's0', reward_dict['b'])
    t5 = Transition('a', 's2', 's3', reward_dict['a'])
    t6 = Transition('b', 's2', 's0', reward_dict['b'])
    t7 = Transition('a', 's3', 's3', reward_dict['a'])
    t8 = Transition('b', 's3', 's3', reward_dict['b'])
    dfa = Process('combination_lock4', states=['s0', 's1', 's2', 's3'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8],
                  initial_state='s0', accepting_states=['s3'])

    return dfa, params

# Parentheses 3 no a
def example25(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 6
        gamma = 0.2  # discount factor
        # reward_dict = {'(': [30, 60, 0], ')': [10, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}
        episode_len = 12
        num_iterations = 1200
        eps = 90
        eps2 = 140
        COMP_LAYER_LEN = 3

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 6  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}

        episode_len = 10
        num_iterations = 800
        eps = 90

        if RPNI:
            episode_len = 7
            num_iterations = 160

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    # t1 = Transition('a', 's0', 's0', reward_dict['a'])
    t2 = Transition('(', 's0', 's1', reward_dict['('])
    t3 = Transition(')', 's0', 's4', reward_dict[')'])
    # t4 = Transition('a', 's1', 's1', reward_dict['a'])
    t5 = Transition('(', 's1', 's2', reward_dict['('])
    t6 = Transition(')', 's1', 's0', reward_dict[')'])
    # t7 = Transition('a', 's2', 's2', reward_dict['a'])
    t8 = Transition('(', 's2', 's3', reward_dict['('])
    t9 = Transition(')', 's2', 's1', reward_dict[')'])
    # t10 = Transition('a', 's3', 's3', reward_dict['a'])
    t11 = Transition('(', 's3', 's4', reward_dict['('])
    t12 = Transition(')', 's3', 's2', reward_dict[')'])
    t13 = Transition('(', 's4', 's4', reward_dict['('])
    t14 = Transition(')', 's4', 's4', reward_dict[')'])

    dfa = Process('parentheses_3', states=['s0', 's1', 's2', 's3', 's4'],
                  transitions=[t2, t3, t5, t6, t8, t9, t11, t12, t13, t14],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params


# Parentheses 4 no a
def example26(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 8
        gamma = 0.2  # discount factor
        # reward_dict = {'(': [30, 60, 0], ')': [10, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}
        episode_len = 16
        num_iterations = 2500
        eps = 82
        eps2 = 120
        COMP_LAYER_LEN = 4

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 8  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}

        episode_len = 13
        num_iterations = 2500
        eps = 100

        if RPNI:
            episode_len = 9
            num_iterations = 400

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('(', 's0', 's1', reward_dict['('])
    t2 = Transition(')', 's0', 's5', reward_dict[')'])

    t3 = Transition('(', 's1', 's2', reward_dict['('])
    t4 = Transition(')', 's1', 's0', reward_dict[')'])

    t5 = Transition('(', 's2', 's3', reward_dict['('])
    t6 = Transition(')', 's2', 's1', reward_dict[')'])

    t7 = Transition('(', 's3', 's4', reward_dict['('])
    t8 = Transition(')', 's3', 's2', reward_dict[')'])

    t9 = Transition('(', 's4', 's5', reward_dict['('])
    t10 = Transition(')', 's4', 's3', reward_dict[')'])

    t11 = Transition('(', 's5', 's5', reward_dict['('])
    t12 = Transition(')', 's5', 's5', reward_dict[')'])

    dfa = Process('parentheses_4', states=['s0', 's1', 's2', 's3', 's4', 's5'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params


# Parentheses 5 no a
def example27(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 8
        gamma = 0.2  # discount factor
        # reward_dict = {'(': [30, 60, 0], ')': [10, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}
        episode_len = 17
        num_iterations = 4000
        eps = 60
        eps2 = 120
        COMP_LAYER_LEN = 4

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 10  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}

        episode_len = 15
        num_iterations = 10000
        eps = 70

        if RPNI:
            episode_len = 11
            num_iterations = 1400

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    # t1 = Transition('a', 's0', 's0', reward_dict['a'])
    t2 = Transition('(', 's0', 's1', reward_dict['('])
    t3 = Transition(')', 's0', 's6', reward_dict[')'])
    # t4 = Transition('a', 's1', 's1', reward_dict['a'])
    t5 = Transition('(', 's1', 's2', reward_dict['('])
    t6 = Transition(')', 's1', 's0', reward_dict[')'])
    # t7 = Transition('a', 's2', 's2', reward_dict['a'])
    t8 = Transition('(', 's2', 's3', reward_dict['('])
    t9 = Transition(')', 's2', 's1', reward_dict[')'])
    # t10 = Transition('a', 's3', 's3', reward_dict['a'])
    t11 = Transition('(', 's3', 's4', reward_dict['('])
    t12 = Transition(')', 's3', 's2', reward_dict[')'])
    # t13 = Transition('a', 's4', 's4', reward_dict['a'])
    t14 = Transition('(', 's4', 's5', reward_dict['('])
    t15 = Transition(')', 's4', 's3', reward_dict[')'])
    # t16 = Transition('a', 's5', 's5', reward_dict['a'])
    t17 = Transition('(', 's5', 's6', reward_dict['('])
    t18 = Transition(')', 's5', 's4', reward_dict[')'])
    # t19 = Transition('a', 's6', 's6', reward_dict['a'])
    t20 = Transition('(', 's6', 's6', reward_dict['('])
    t21 = Transition(')', 's6', 's6', reward_dict[')'])

    # dfa = Process('parentheses_5', states=['s0', 's1', 's2', 's3', 's4', 's5', 's5', 's6'],
    #               transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21],
    #               initial_state='s0', accepting_states=['s0'])
    dfa = Process('parentheses_5', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6'],
                  transitions=[t2, t3, t5, t6, t8, t9, t11, t12, t14, t15, t17, t18,
                               t20, t21],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params

# Parentheses 6 no a
def example29(RPNI=False, Dual=False):
    if Dual:
        # general parameters
        STATE_LEN = 8
        gamma = 0.2  # discount factor
        # reward_dict = {'(': [30, 60, 0], ')': [10, 20, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}
        episode_len = 19
        num_iterations = 17000
        eps = 38
        eps2 = 130
        COMP_LAYER_LEN = 6

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 11  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'(': [20, 1200, 0], ')': [400, 800, 0]}

        episode_len = 19
        num_iterations = 17000
        eps = 35

        if RPNI:
            episode_len = 17
            num_iterations = 1000

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('(', 's0', 's1', reward_dict['('])
    t2 = Transition(')', 's0', 's7', reward_dict[')'])

    t3 = Transition('(', 's1', 's2', reward_dict['('])
    t4 = Transition(')', 's1', 's0', reward_dict[')'])

    t5 = Transition('(', 's2', 's3', reward_dict['('])
    t6 = Transition(')', 's2', 's1', reward_dict[')'])

    t7 = Transition('(', 's3', 's4', reward_dict['('])
    t8 = Transition(')', 's3', 's2', reward_dict[')'])

    t9 = Transition('(', 's4', 's5', reward_dict['('])
    t10 = Transition(')', 's4', 's3', reward_dict[')'])

    t11 = Transition('(', 's5', 's6', reward_dict['('])
    t12 = Transition(')', 's5', 's4', reward_dict[')'])

    t13 = Transition('(', 's6', 's7', reward_dict['('])
    t14 = Transition(')', 's6', 's5', reward_dict[')'])

    t15 = Transition('(', 's7', 's7', reward_dict['('])
    t16 = Transition(')', 's7', 's7', reward_dict[')'])

    dfa = Process('parentheses_6', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params

# Divisibility by 4
# (0|(10|111*0)(10|111*0)*0)*
def example34(RPNI=False, Dual=False):
    if Dual:
        STATE_LEN = 4
        # general parameters
        # gamma = 0.6  # discount factor
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        # reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 8
        num_iterations = 50
        eps = 50
        eps2 = 100
        COMP_LAYER_LEN = 2

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 5  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        episode_len = 7
        num_iterations = 50
        eps = 50

        if RPNI:
            episode_len = 4
            num_iterations = 35

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('0', 's0', 's0', reward_dict['0'])
    t2 = Transition('1', 's0', 's1', reward_dict['1'])
    t3 = Transition('0', 's1', 's2', reward_dict['0'])
    t4 = Transition('1', 's1', 's1', reward_dict['1'])
    t5 = Transition('0', 's2', 's0', reward_dict['0'])
    t6 = Transition('1', 's2', 's1', reward_dict['1'])

    dfa = Process('Divisibility_by_4', states=['s0', 's1', 's2'],
                  transitions=[t1, t2, t3, t4, t5, t6],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params


# Divisibility by 5
# grep syntax: (0|1((10)*(0|11)(01*01)*01*00)*(10)*(0|11)(01*01)*1)*
def example35(RPNI=False, Dual=False):
    if Dual:
        STATE_LEN = 5
        # general parameters
        # gamma = 0.6  # discount factor
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        # reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 10
        num_iterations = 180
        eps = 30
        eps2 = 40
        COMP_LAYER_LEN = 3

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 5  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        episode_len = 8
        num_iterations = 180
        eps = 50

        if RPNI:
            episode_len = 7
            num_iterations = 200

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('0', 's0', 's0', reward_dict['0'])
    t2 = Transition('1', 's0', 's1', reward_dict['1'])

    t3 = Transition('0', 's1', 's2', reward_dict['0'])
    t4 = Transition('1', 's1', 's3', reward_dict['1'])

    t5 = Transition('0', 's2', 's4', reward_dict['0'])
    t6 = Transition('1', 's2', 's0', reward_dict['1'])

    t7 = Transition('0', 's3', 's1', reward_dict['0'])
    t8 = Transition('1', 's3', 's2', reward_dict['1'])

    t9 = Transition('0', 's4', 's3', reward_dict['0'])
    t10 = Transition('1', 's4', 's4', reward_dict['1'])

    dfa = Process('Divisibility_by_5', states=['s0', 's1', 's2', 's3', 's4'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params


# Divisibility by 6
#
# grep syntax: (0|(11|1(00|011*0)(00|011*0)*1)(11|1(00|011*0)(00|011*0)*1)*0)*
def example36(RPNI=False, Dual=False):
    if Dual:
        STATE_LEN = 5
        # general parameters
        # gamma = 0.6  # discount factor
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        # reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 10
        num_iterations = 80
        eps = 35
        eps2 = 100
        COMP_LAYER_LEN = 3

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 5  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}

        episode_len = 8
        num_iterations = 100
        eps = 35

        if RPNI:
            episode_len = 6
            num_iterations = 80

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('0', 's0', 's0', reward_dict['0'])
    t2 = Transition('1', 's0', 's1', reward_dict['1'])

    t3 = Transition('0', 's1', 's2', reward_dict['0'])
    t4 = Transition('1', 's1', 's3', reward_dict['1'])

    t5 = Transition('0', 's2', 's4', reward_dict['0'])
    t6 = Transition('1', 's2', 's2', reward_dict['1'])

    t7 = Transition('0', 's3', 's0', reward_dict['0'])
    t8 = Transition('1', 's3', 's1', reward_dict['1'])

    t9 = Transition('0', 's4', 's2', reward_dict['0'])
    t10 = Transition('1', 's4', 's3', reward_dict['1'])

    dfa = Process('Divisibility_by_6', states=['s0', 's1', 's2', 's3', 's4'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params


# Divisibility by 7
def example37(RPNI=False, Dual=False):
    if Dual:
        STATE_LEN = 7
        # general parameters
        # gamma = 0.6  # discount factor
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        # reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 14
        num_iterations = 700
        eps = 19
        eps2 = 40
        COMP_LAYER_LEN = 4

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 7  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}

        episode_len = 11
        num_iterations = 600
        eps = 20

        if RPNI:
            episode_len = 8
            num_iterations = 80

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('0', 's0', 's0', reward_dict['0'])
    t2 = Transition('1', 's0', 's1', reward_dict['1'])

    t3 = Transition('0', 's1', 's2', reward_dict['0'])
    t4 = Transition('1', 's1', 's3', reward_dict['1'])

    t5 = Transition('0', 's2', 's4', reward_dict['0'])
    t6 = Transition('1', 's2', 's5', reward_dict['1'])

    t7 = Transition('0', 's3', 's6', reward_dict['0'])
    t8 = Transition('1', 's3', 's0', reward_dict['1'])

    t9 = Transition('0', 's4', 's1', reward_dict['0'])
    t10 = Transition('1', 's4', 's2', reward_dict['1'])

    t11 = Transition('0', 's5', 's3', reward_dict['0'])
    t12 = Transition('1', 's5', 's4', reward_dict['1'])

    t13 = Transition('0', 's6', 's5', reward_dict['0'])
    t14 = Transition('1', 's6', 's6', reward_dict['1'])

    dfa = Process('Divisibility_by_7', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params

# Divisibility by 9
def example40(RPNI=False, Dual=False):
    if Dual:
        STATE_LEN = 6
        # general parameters
        # gamma = 0.6  # discount factor
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}
        # reward_dict = {'a': [20, 100, 0], 'b': [40, 60, 0]}
        # reward_dict = {'a': [20, 300, 0], 'b': [120, 240, 0]}
        episode_len = 12
        num_iterations = 750
        eps = 21
        eps2 = 40
        COMP_LAYER_LEN = 3

        params = GeneralParameters(COMP_LAYER_LEN=COMP_LAYER_LEN,
                                   STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   eps2=eps2)

    else:
        STATE_LEN = 6  # longest prefix in memory
        gamma = 0.2  # discount factor
        reward_dict = {'0': [20, 300, 0], '1': [120, 240, 0]}

        episode_len = 10
        num_iterations = 1000
        eps = 23

        if RPNI:
            episode_len = 10
            num_iterations = 80

        params = GeneralParameters(STATE_LEN=STATE_LEN,
                                   gamma=gamma,
                                   reward_dict=reward_dict,
                                   episode_len=episode_len,
                                   num_iterations=num_iterations,
                                   eps=eps)

    t1 = Transition('0', 's0', 's0', reward_dict['0'])
    t2 = Transition('1', 's0', 's1', reward_dict['1'])

    t3 = Transition('0', 's1', 's2', reward_dict['0'])
    t4 = Transition('1', 's1', 's3', reward_dict['1'])

    t5 = Transition('0', 's2', 's4', reward_dict['0'])
    t6 = Transition('1', 's2', 's5', reward_dict['1'])

    t7 = Transition('0', 's3', 's6', reward_dict['0'])
    t8 = Transition('1', 's3', 's7', reward_dict['1'])

    t9 = Transition('0', 's4', 's8', reward_dict['0'])
    t10 = Transition('1', 's4', 's0', reward_dict['1'])

    t11 = Transition('0', 's5', 's1', reward_dict['0'])
    t12 = Transition('1', 's5', 's2', reward_dict['1'])

    t13 = Transition('0', 's6', 's3', reward_dict['0'])
    t14 = Transition('1', 's6', 's4', reward_dict['1'])

    t15 = Transition('0', 's7', 's5', reward_dict['0'])
    t16 = Transition('1', 's7', 's6', reward_dict['1'])

    t17 = Transition('0', 's8', 's7', reward_dict['0'])
    t18 = Transition('1', 's8', 's8', reward_dict['1'])

    dfa = Process('Divisibility_by_9', states=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'],
                  transitions=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18],
                  initial_state='s0', accepting_states=['s0'])

    return dfa, params
