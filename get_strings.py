# generates all the possible strings over a given alphabet, up to length k
def get_up_to_k_length(set, k):
    str_lst = [""]
    for l in range(1, k+1):
        str_lst.extend(get_all_k_length(set, l))

    return str_lst

# generates all the possible strings of length k over a given alphabet
def get_all_k_length(set, k):
    n = len(set)
    str_lst = []
    get_all_k_length_rec(set, "", n, k, str_lst)

    return str_lst


# The main recursive method to print all possible strings of length k
def get_all_k_length_rec(set, prefix, n, k, str_lst):
    # Base case: k is 0,
    if (k == 0):
        str_lst.append(prefix)
        return

    # One by one add all characters from set and recursively call for k equals to k-1
    for i in range(n):
        # Next character of input added
        newPrefix = prefix + set[i]

        # k is decreased, because
        # we have added a new character
        get_all_k_length_rec(set, newPrefix, n, k - 1, str_lst)