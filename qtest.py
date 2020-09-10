import queue
import _thread
from collections import deque
import torch.nn as nn


class Cifar(nn.Module):
    def __init__(self):
        super(Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, bias=False)


def test():
    q = deque()
    for i in range(100):
        q.append(i)

    while True:
        try:
            print(q.popleft())
        except:
            break

    print(q)


def test2():
    q = [i for i in range(100)]
    for i in q:
        print(i)


def test3():
    q = deque()
    print(not q)


if __name__ == '__main__':
    test3()
