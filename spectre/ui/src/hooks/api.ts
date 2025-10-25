"use client";
import { parse } from "csv-parse";
import useSWR from "swr";

function parse_replay(data: string): ReplayRow[] {
    // Parse pipe-separated dsv. Each line is <timestamp>|<log>. If log contains a pipe, cell will be in quotes.
    const contents = [];
    const parser = parse(
        {
            delimiter: "|",
            columns: ["timestamp", "log"],
        },
        (err, records) => {
            if (err) throw err;
            return records;
        },
    );

    parser.write(data);
    let record: { timestamp: string; log: string };
    // biome-ignore lint/suspicious/noAssignInExpressions: From csv-parse docs
    while ((record = parser.read()) !== null) {
        contents.push({
            timestamp: new Date(+record.timestamp),
            log: record.log,
        });
    }

    return contents;
}

interface ReplayRow {
    timestamp: Date;
    log: string;
}

const fetcher = (url: string) =>
    fetch(url)
        .then((res) => res.text())
        .then((data) => parse_replay(data));

export function useReplayData(id: string) {
    const { data, error } = useSWR(
        `${process.env.NEXT_PUBLIC_API_URL}/display/${id}`,
        fetcher,
        { refreshInterval: 100 }
    );

    return {
        data,
        isLoading: !error && !data,
        isError: error,
    };
}
