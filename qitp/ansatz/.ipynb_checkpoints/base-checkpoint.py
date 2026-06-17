from abc import ABC, abstractmethod


class OperatorBasis(ABC):

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def matrix(self, i):
        pass

    @abstractmethod
    def product(self, i, j):
        pass

    @abstractmethod
    def operator(self, i):
        pass

    @abstractmethod
    def from_seed(hamiltonian, seeds, depth):
        pass