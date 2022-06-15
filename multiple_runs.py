from utility_based import *


def repeat_experiment(repetitions):
    avg_time, avg_samples_count, success_count = 0, 0, 0

    for _ in range(repetitions):
        Dual = False
        RPNI = False
        dfa, params = example9(RPNI=RPNI, Dual=Dual)
        con_dfa, acc_prefixes, rej_prefixes, time_elapsed = single_run(dfa=dfa, params=params, Dual=Dual)

        if con_dfa.is_pos_neg_consistent(acc_prefixes, rej_prefixes) and len(dfa.states) == len(con_dfa.states):
            print("success")
            avg_time += time_elapsed
            avg_samples_count += (len(acc_prefixes) + len(rej_prefixes))
            success_count += 1
        else:
            print("failed")

    print("\n\nResults for experiment: {}, {} repetitions".format(dfa.name, repetitions))
    print("Success rate: {}%".format((success_count/repetitions)*100))
    print("Average number of required samples: {}".format(avg_samples_count/success_count))
    print("Average time: {} s".format(avg_time/success_count))


if __name__ == '__main__':
    repeat_experiment(10)
