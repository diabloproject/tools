### Networking
Hierarchy:
- Namespace
- Task
- Instance

Namespaces are logical divisions of tasks, allowing them to access same resources.
For example, when lauching a CI task, you probably want every task to have access to your build cache.
Those namespaces also share the same digits in address (before the first colon).
Next level of organisation is a task. Tasks are individual runs of something, scheduled together.
And the final level is an instance: When running something like mapreduce or a python webserver, multiple processes can be attributed to the same task.
In that case, instance field in the address will help you differenciate them, as long as all processes were launched by the tailborn runtime.
```
addr = 0000:0000:00000000
        ^     ^      ^
    namespace |      |
             task    |
                   instance
```
### Control server
Control server acts like a mediator between every other node, including (but not limited to):
- Transferring all the data between the nodes.
- Keeping track of task instances and routing data between them
