from prefsampling.approval import impartial


def validate_or_generate_central_vote(
    num_candidates,
    rel_size_central_vote,
    central_vote=None,
    impartial_central_vote=False,
    seed=None,
):
    """
    Validates or generates a central vote based on the different parameters.
    """
    k = int(rel_size_central_vote * num_candidates)
    if impartial_central_vote:
        central_vote = impartial(1, num_candidates, rel_size_central_vote, seed=seed)[0]
    else:
        if central_vote:
            if not all(int(c) == c for c in central_vote):
                raise ValueError(
                    f"The central vote needs to be a set of int (current value is"
                    f" {central_vote} of type {type(central_vote)}["
                    f"{type(next(iter(central_vote)))}])."
                )
            if max(central_vote) > num_candidates - 1 or min(central_vote) < 0:
                raise ValueError(
                    "The elements of the central vote cannot be smaller than 0 "
                    f"(min is currently {min(central_vote)}) and cannot be larger "
                    f"than {num_candidates - 1} (max is currently "
                    f"{max(central_vote)})."
                )
            central_vote = set(int(c) for c in central_vote)
        else:
            central_vote = set(range(k))
    return central_vote
