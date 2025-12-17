"""Multi-stage training"""
from abc import ABC, abstractmethod


class Stage[ModelT](ABC):
    model: ModelT

    @abstractmethod
    def transition(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def train(self):
        pass


class MstStageExecutor:
    stages: list[Stage]

    def __init__(self, stages) -> None:
        self.stages = stages

    def run(self):
        pass
