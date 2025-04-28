This project implements `tb` command, responsible for actual distributed compute.

Let's look at the command at hand:

```bash
cat ./input.csv | csv2tbh | tb python "@map.py" | tdh2csv > ./output.csv
```
- `cat ./input.csv` — we are reading input.csv into stdout, then piping it into stdin of the next command
- `csv2tdh` — this utility translates csv input into .tdh (tailborn DFS Handles, large integers pointing to internal DSF objects, containing corresponding rows as files in a special format). There are other utilities, like `jsonl2tdh`, but we can ignore them for now. csv in stdin, `\n`-separated list of ints to stdout
- `tb python @map.py` — our goal. This takes list of tdh handles in stdin, then executes command `python map.py` as many times as cores in the CPU, piping rows to them, and getting the output rows. For example, if our CPU has 2 cores and there are 4 rows, 2 total commands will recieve two rows each, in parallel. The @ sign is used to reference the file from the current working directory, sending it to the tailborn DFS. the command placed after `tb` is executed by internal shell, meaning that arguments provided will be passed as is, except for @ sign for input file, and & for output files.
- `tdh2csv` — see `csv2tdh`, but backwards.

## DFS
Tailborn DFS can store any content, that can be represented in byte sequence. We refer to those pieces of content as *objects*.
You do not need to write an actual implementation of dfs client. Just write anything that conforms to DfsClient trait, and I will use my library to write an actual implmentation. Just be careful to use the DI properly, so I can just plug it in.

## Project structure
Write something that actually makes sense and has good and resiliant architecture. Be careful to not depend on linux: This project have to run on windows and mac also.
You are free to add dependencies that make your life easier, but you will need to explain why you need that dependency.
More files >>> More lines in a single file.
Write error types, instead of relying on anyhow. Anyhow is only allowed in the main file. Every error has to implement Error trait, you can use thiserror for simplicity.
Add necessary docs, but do not add useless comments. Comments should explain behaviour, not describe it.

Good luck!
