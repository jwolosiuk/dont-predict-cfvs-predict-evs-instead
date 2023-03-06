import torch
from utils import board_int_to_board_str, device

buckets_per_board = torch.load('data/buckets_per_board.pt')  # dict: board string -> tensor of shape (1326,)
# buckets are represented as tensors: for each hand, index of an assigned bucket
n_buckets = 1000


def bucketization_of_ranges(ranges_batch, boards_batch):
    """ performs bucketization of ranges = sum of range for hands in a bucket """
    bucketed_ranges_batch = []
    for ranges, board in zip(ranges_batch, boards_batch):
        board = board_int_to_board_str(board)
        buckets = buckets_per_board[board]
        bucketed_ranges = []
        for p in ranges:
            bucketed_ranges.append(torch.scatter(torch.zeros(n_buckets).float(), 0, buckets, p.cpu(), reduce='add'))
        bucketed_ranges_batch.append(torch.stack(bucketed_ranges))
    bucketed_ranges_batch = torch.stack(bucketed_ranges_batch)
    return bucketed_ranges_batch.to(device)


def bucketization_of_values(values_batch, boards_batch):
    """ performs bucketization of cfvs/EVs = mean of values for hands in a bucket """
    bucketed_values_batch = []
    for values, board in zip(values_batch, boards_batch):
        board = board_int_to_board_str(board)
        buckets = buckets_per_board[board]
        n_hands_in_bucket = torch.scatter(torch.zeros(n_buckets).float(), 0, buckets, torch.ones_like(buckets).float(),
                                          reduce='add')
        sum_values_in_bucket = []
        for p in values:
            sum_values_in_bucket.append(torch.scatter(torch.zeros(n_buckets).float(), 0, buckets, p.cpu(), reduce='add'))
        sum_values_in_bucket = torch.stack(sum_values_in_bucket)
        bucketed_values = sum_values_in_bucket / n_hands_in_bucket
        bucketed_values_batch.append(bucketed_values)
    bucketed_values_batch = torch.stack(bucketed_values_batch)
    return bucketed_values_batch.to(device)


def unbucketization(bucketed_values_batch, boards_batch):
    """ performs unbucketization = copy values for hands in a buckets """
    unbucketed_bucketed_values_batch = []
    for buck_vals, board in zip(bucketed_values_batch, boards_batch):
        board = board_int_to_board_str(board)
        buckets = buckets_per_board[board]
        unbuck_buck_vals = []
        for p in buck_vals:
            unbuck_buck_vals.append(p.cpu().gather(0, buckets))
        unbucketed_bucketed_values_batch.append(torch.stack(unbuck_buck_vals))
    unbucketed_bucketed_values_batch = torch.stack(unbucketed_bucketed_values_batch)
    return unbucketed_bucketed_values_batch.to(device)