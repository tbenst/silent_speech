##
import random, numpy as np
from collections import defaultdict
from typing import List, Tuple


def pop_next_class_from_items(items, my_class):
    """
    Given a list of items, return the index of the first item that matches
    my_class, removing it from the list
    """
    for index, item_class in enumerate(items):
        if item_class == my_class:
            return index, items.pop(index), items
    return None, None, items


assert pop_next_class_from_items([1, 4, 2, 2, 3], 2) == (2, 2, [1, 4, 2, 3])
assert pop_next_class_from_items([1, 2, 3], 5) == (None, None, [1, 2, 3])


##
def fill_remaining_bins(
    bins, bin_sums, idx_per_class, lengths, max_len, class_proportion, class_debt
):
    "Try to fill remaining bins with random classes"
    rejected_sample_count = 0

    # keep trying until 100 consecutive rejections
    while rejected_sample_count < 100:
        # randomly sample remaining classes with indices
        valid_classes = [c for c in idx_per_class if len(idx_per_class[c]) > 0]
        if len(valid_classes) == 0:
            break
        # adjust proportion for remaining classes
        proportion = np.array([class_proportion[c] for c in valid_classes])
        proportion /= proportion.sum()

        sampled_class = np.random.choice(valid_classes, p=proportion)

        # if we can pay off debt, do so and sample again
        if class_debt[sampled_class] > 0:
            class_debt[sampled_class] -= 1
            rejected_sample_count = 0
            continue

        # try to pack sampled_length into a bin
        added, idx_per_class, bins, bin_sums = add_class_to_any_bin(
            sampled_class, idx_per_class, bins, bin_sums, lengths, max_len
        )
        if added:
            rejected_sample_count = 0
        else:
            rejected_sample_count += 1
            continue

    return bins


def add_class_to_bin(c, bin_idx, idx_per_class, bins, bin_sums, lengths, max_len):
    "pop last index of class c, adding to bin, add length to bin_sums"
    if len(idx_per_class[c]) == 0:
        raise ValueError(f"no more items of class {c} to pack")
    length = lengths[idx_per_class[c][-1]]
    if bin_sums[bin_idx] + length > max_len:
        # can't fit this item, stop packing bins
        return False, idx_per_class, bins, bin_sums
    bin_sums[bin_idx] += lengths[idx_per_class[c][-1]]
    bins[bin_idx].append(idx_per_class[c].pop())
    return True, idx_per_class, bins, bin_sums


def test_add_class_to_bin():
    bins = [[]]
    idx_per_class = {"A": [0, 1, 2, 3], "B": []}
    bin_sums = [0]
    lengths = [4, 1, 4, 4]
    max_len = 5

    # Normal case
    success, idx_per_class, bins, bin_sums = add_class_to_bin(
        "A", 0, idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success
    assert bins == [[3]]

    # Exceeding max length
    success, idx_per_class, bins, bin_sums = add_class_to_bin(
        "A", 0, idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert not success

    # Empty class list
    try:
        add_class_to_bin("B", 0, idx_per_class, bins, bin_sums, lengths, max_len)
    except ValueError as e:
        assert str(e) == "no more items of class B to pack"


test_add_class_to_bin()


def add_class_to_any_bin(c, idx_per_class, bins, bin_sums, lengths, max_len):
    "try to pack class into a bin"
    success = False  # needed for no bins case
    for i in range(len(bins)):
        success, idx_per_class, bins, bin_sums = add_class_to_bin(
            c, i, idx_per_class, bins, bin_sums, lengths, max_len
        )
        if success:
            break
    return success, idx_per_class, bins, bin_sums


def test_add_class_to_any_bin():
    lengths = [3, 5, 2, 2, 1]
    max_len = 5
    idx_per_class = {"A": [0, 1], "B": [2, 3, 4]}
    bins = [[], []]
    bin_sums = [0, 0]

    # Test where addition is successful
    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "A", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success

    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "B", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success

    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "A", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success

    assert bins[0] == [1]
    assert bin_sums[0] == 5
    assert bins[1] == [4, 0]
    assert bin_sums[1] == 4

    # Test where addition is not possible due to size constraints
    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "B", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert not success

    assert bins[0] == [1]
    assert bin_sums[0] == 5
    assert bins[1] == [4, 0]
    assert bin_sums[1] == 4


test_add_class_to_any_bin()


def pack_items(
    lengths: List[int],
    classes: List[int],
    max_len: int,
    class_proportion: Tuple[float],
    always_include: List[int],
    shuffle: bool = True,
):
    """
    Greedily and randomly packs each of the N items into a bin (list) of
    max_len, returning a list of bins containing the index of the pertinent
    items. Stops when no more items of a class are left to pack.

    If a class in always_include is low weighting in class_proportion, and
    max_len is small,
    """
    # drop any index that exceeds max_len
    items = []
    new_classes = []
    for i, l in enumerate(lengths):
        if l <= max_len:
            items.append(i)
            new_classes.append(classes[i])

    # group indices by class
    idx_per_class = defaultdict(list)
    for i, c in zip(items, new_classes):
        idx_per_class[c].append(i)

    # shuffle indices within each class
    if shuffle:
        for c in idx_per_class:
            random.shuffle(idx_per_class[c])

    valid_classes = np.array(list(set(idx_per_class.keys())))
    assert len(valid_classes) > 0, "no items to pack under max_len"
    bins = []
    bin_sums = []

    # when creating a new bin, since we are always including some classes,
    # we need to compensate for the imbalance in class_proportion
    # by ignoring the random choice until the debt is paid
    class_debt = {c: 0 for c in valid_classes}

    while True:
        # print("DEBUG: start while loop")
        # print(f"DEBUG: {idx_per_class=}, {bins=}")
        stop_loop = False
        for v in idx_per_class.values():
            if len(v) == 0:
                stop_loop = True
        if stop_loop:
            break
        sampled_class = np.random.choice(valid_classes, p=class_proportion)

        # check if we can pay off debt
        if class_debt[sampled_class] > 0:
            class_debt[sampled_class] -= 1
            continue

        # try to pack sampled_length into a bin
        added, idx_per_class, bins, bin_sums = add_class_to_any_bin(
            sampled_class, idx_per_class, bins, bin_sums, lengths, max_len
        )

        if not added:
            # create a new bin
            bins.append([])
            bin_sums.append(0)
            # add required classes to bin
            # print(f"DEBUG: {idx_per_class=}")
            for c in always_include:
                # print(f"DEBUG: {c=}, {bins=}")
                success, idx_per_class, bins, bin_sums = add_class_to_bin(
                    c, -1, idx_per_class, bins, bin_sums, lengths, max_len
                )
                if not success:
                    # print(f"DEBUG: {idx_per_class=}, {bins=}")
                    raise ValueError(
                        "hit the unusual edge case that is slightly annoying but solvable"
                    )
                    # unable to add any more valid bins with required classes
                    # TODO
                    # EDGE CASE: if we fail to make a valid bin because of bad luck
                    # eg if always_include = [1, 2] and 1 & 2 don't fit together
                    return fill_remaining_bins(
                        bins[:-1],
                        idx_per_class,
                        lengths,
                        max_len,
                        class_proportion,
                        class_debt,
                    )
                if c == sampled_class:
                    # no debt accrues
                    added = True
                    continue
                else:
                    # debt accrues
                    class_debt[c] += 1

        # if the sampled class is not part of always_include, we need to fit it somewhere
        if not added:
            assert (
                sampled_class not in always_include
            ), f"{sampled_class} in always_include"
            # try to pack sampled_length into new bin
            success, idx_per_class, bins, bin_sums = add_class_to_bin(
                sampled_class, -1, idx_per_class, bins, bin_sums, lengths, max_len
            )
            if not success:
                # either this item is annoyingly long, or we got an unlucky
                # draw of new bin. can't pack this item, so push to back
                index = idx_per_class[sampled_class].pop()
                idx_per_class[sampled_class].insert(0, index)

    return fill_remaining_bins(
        bins, bin_sums, idx_per_class, lengths, max_len, class_proportion, class_debt
    )


def test_complex_pack_items(
    lengths,
    classes,
    class_proportion,
    always_include,
    max_len,
    max_proportion_error=0.05,
):
    # complex example that mimics gaddy dataset
    N = len(lengths)
    bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
    # check class proportions
    for i in range(3):
        sampled_proportion = np.mean(
            [classes[item] == i for bin in bins for item in bin]
        )
        print(f"{sampled_proportion=}, {class_proportion[i]=}")
        assert (
            abs(sampled_proportion - class_proportion[i]) < max_proportion_error
        ), f"{sampled_proportion=}, {class_proportion[i]=}"

    for bin in bins:
        assert 0 in [classes[item] for item in bin]
        assert 1 in [classes[item] for item in bin]
        assert sum(lengths[item] for item in bin) <= max_len

    bin_classes = [[classes[item] for item in bin] for bin in bins]
    bin_lengths = [sum(lengths[item] for item in bin) for bin in bins]
    N_packed = sum(len(bin) for bin in bins)
    print(f"{bin_classes=}")
    print(f"{bin_lengths=}")
    print(f"{N_packed=}, {N=}, {min(bin_lengths)=}")
    assert (
        min(bin_lengths) >= max_len * 0.85
    ), f"{min(bin_lengths)=}, is too small vs {max_len=}"


def test_pack_items():
    lengths = [2, 3, 1, 4]
    classes = [0, 0, 1, 1]
    max_len = 4
    class_proportion = (0.5, 0.5)
    always_include = [1]

    # Normal case
    bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
    assert all(
        sum(lengths[item] for item in bin) <= max_len for bin in bins
    )  # All bins must respect max_len
    for bin in bins:
        classes_in_bin = [classes[item] for item in bin]
        assert 1 in classes_in_bin, f"All bins must include class 1 {bins=}"

    # I think we may be done..? maybe need a couple more tests

    # Test with items larger than max_len
    lengths = [5, 6, 7]
    classes = [1, 1, 1]
    bins = pack_items(lengths, classes, 5, [1.0], always_include)
    assert len(bins) == 1  # one items should be packed

    # Test with empty class list
    lengths = []
    classes = []
    try:
        bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
    except AssertionError as e:
        assert str(e) == "no items to pack under max_len"

    N = 1000
    test_complex_pack_items(
        lengths=np.random.randint(1, 20, N),
        classes=np.random.randint(0, 3, N),
        class_proportion=[0.08, 0.42, 0.5],
        always_include=[0, 1],
        max_len=100,
    )

    test_complex_pack_items(
        lengths=np.random.randint(1, 20, N),
        # due to these probabilities, we may not pack many bins since we run out of class 0
        classes=np.random.choice([0, 1, 2], N, p=[0.01, 0.09, 0.9]),
        class_proportion=[0.08, 0.42, 0.5],
        always_include=[0, 1],
        max_len=100,
        max_proportion_error=0.1 # we don't always nail the proportions, but are close enough
    )


test_pack_items()

# # Example usage
# lengths = [2, 3, 4, 5, 1, 3, 2]
# classes = [1, 2, 3, 1, 2, 3, 1]
# max_len = 5
# class_proportion = [0.33, 0.33, 0.34]  # example proportions
# always_include = [1, 2]

# packed_bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
# print(packed_bins)

##
