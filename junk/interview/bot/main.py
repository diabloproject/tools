#!/usr/bin/env python3
from dataclasses import dataclass



polls: dict[str, list[str]] = {
    "poll 1": ["q1", "q2", "q3"],
    "poll 2": ["q1", "q2", "q3"],
    "poll 3": ["q1", "q2", "q3"]
}


@dataclass
class State:
    current_poll: str = 'entry'
    current_question: int | None = None

    def next(self, input) -> 'State':
        if input == '/start':
            print("Добро пожаловать")
            return State(current_poll="free")
        match self.current_poll:
            case "entry":
                return self
            case "free":
                for p in polls:
                    if ('/' + p) == input:
                        print(polls[p][0])
                        return State(current_poll=p, current_question=1)
                print("Invalid poll")
                return self
            case _:
                poll = polls[self.current_poll]
                if len(poll) == self.current_question:
                    print("Завершено")
                    return State(current_poll="free", current_question=None)
                else:
                    assert self.current_question is not None
                    print(polls[self.current_poll][self.current_question])
                    return State(current_poll=self.current_poll, current_question=self.current_question + 1)


def main():
    state = State(current_poll="entry")
    while True:
        i = input()
        state = state.next(i)


if __name__ == "__main__":
    main()
