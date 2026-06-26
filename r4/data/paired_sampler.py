from __future__ import annotations

from itertools import cycle

from torch.utils.data import DataLoader


def paired_batches(labeled_loader, unlabeled_loader):
    for batch_l, batch_u in zip(cycle(labeled_loader), cycle(unlabeled_loader)):
        yield batch_l, batch_u


class InfiniteSemiIterator:
    """Compatibility iterator that continuously yields labeled/unlabeled pairs."""

    def __init__(self, labeled_loader, unlabeled_loader):
        self._iterator = paired_batches(labeled_loader, unlabeled_loader)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)


def make_loader(dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, drop_last: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
