import torch
from utils import HANDS, split_into_cards, board_int_to_board_str, device


def block_matrix_for_board(board):
    """ for a given board creates 0-1 matrix, which opponent hands are blocked by board or hero hand """
    blocked_matrix = []
    for h1 in HANDS:
        blocked_cards = split_into_cards(board+h1)
        hands_blocked_by_h1_or_board = []
        for h2 in HANDS:
            is_h2_blocked = any([b in h2 for b in blocked_cards])
            hands_blocked_by_h1_or_board.append(is_h2_blocked)
        hands_blocked_by_h1_or_board = torch.tensor(hands_blocked_by_h1_or_board)
        blocked_matrix.append(hands_blocked_by_h1_or_board)
    blocked_matrix = torch.stack(blocked_matrix)
    return blocked_matrix


def get_matchups(ranges_batch, boards_batch):
    """ for given ranges returns matchups = [sum of not blocked opponent's range for a given hero hand] """
    matchups = []
    for ranges, board in zip(ranges_batch, boards_batch):
        board = board_int_to_board_str(board)
        non_block_matrix = (~block_matrix_for_board(board)).float().to(device)
        match = ranges @ non_block_matrix
        matchups.append(match)
    matchups = torch.stack(matchups)
    return matchups
