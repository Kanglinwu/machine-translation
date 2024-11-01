from itertools import zip_longest

def interleave_lists(list1, list2):
    # Use zip_longest to interleave the lists, filling in None for missing elements
    interleaved = [item for pair in zip_longest(list1, list2) for item in pair if item is not None]
    
    # Join the interleaved list into a single string
    result_string = ' '.join(interleaved)
    
    return result_string