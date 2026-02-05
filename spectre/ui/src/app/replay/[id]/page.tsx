"use client";
import { useReplayData } from "@/hooks/api";
import { use } from "react";
import { List, Text } from "@gravity-ui/uikit";

export default function Page({ params }: { params: Promise<{ id: string }> }) {
    const { id } = use(params);
    const { data, isError } = useReplayData(id);
    if (isError) {
        return <div>Error loading replay</div>;
    }
    if (data === undefined) {
        return <div>Loading...</div>;
    }

    return (
        <div className="w-screen h-screen flex flex-col">
            <List
                items={data}
                renderItem={(item) => {
                    return (
                        <Text variant="code-1" className="select-text">
                            <pre>{item.log}</pre>
                        </Text>
                    );
                }}
                itemHeight={(_) => 16}
                filterable={false}
            />
        </div>
    );
}
