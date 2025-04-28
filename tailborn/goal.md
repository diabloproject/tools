This project implements `tb` command, responsible for actual distributed compute.


Let's look at the command at hand:

```bash
cat ./input.csv | csv2tbh | tb python "@map.py" | tdh2csv > ./output.csv
```
- `cat ./input.csv` — we are reading input.csv into stdout, then piping it into stdin of the next command
- `csv2tdh` — this utility translates csv input into .tdh (tailborn DFS Handles, large integers pointing to internal DSF objects, containing corresponding rows as files in a special format). There are other utilities, like `jsonl2tdh`, but we can ignore them for now. csv in stdin, `\n`-separated list of ints to stdout
- `tb python @map.py` — our goal. This takes list of tdh handles in stdin, then executes command `python map.py` as many times as cores in the CPU, piping rows to them, and getting the output rows. For example, if our CPU has 2 cores and there are 4 rows, 2 total commands will recieve two rows each, in parallel. The @ sign is used to reference the file from the current working directory, sending it to the tailborn DFS. the command placed after `tb` is executed by internal shell, meaning that arguments provided will be passed as is, except for @ sign for input file, and & for output files.
- `tdh2csv` — see `csv2tdh`, btu backwards
