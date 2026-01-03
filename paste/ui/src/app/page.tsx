"use client";

import { List, PlaceholderContainer, Text } from "@gravity-ui/uikit";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface Paste {
    code: string;
    createdAt: string;
    preview: string;
}

export default function Home() {
    const [pastes, setPastes] = useState<Paste[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        async function fetchRecentPastes() {
            try {
                const res = await fetch("/api/paste", {
                    cache: "no-store",
                });

                if (!res.ok) {
                    setPastes([]);
                    return;
                }

                const data = await res.json();
                setPastes(data.pastes || []);
            } catch (error) {
                console.error("Error fetching pastes:", error);
                setPastes([]);
            } finally {
                setLoading(false);
            }
        }

        fetchRecentPastes();
    }, []);

    if (loading) {
        return (
            <div className="p-8">
                <Text variant="header-2" className="mb-6">
                    Recent Pastes
                </Text>
                <Text>Loading...</Text>
            </div>
        );
    }

    return (
        <div className="p-8 flex flex-col h-full">
            <Text variant="header-2" className="mb-6">
                Recent Pastes
            </Text>
            <div className="flex-1 h-full">
                <List
                    items={pastes}
                    filterable={true}
                    virtualized={false}
                    autoFocus
                    emptyPlaceholder={
                        <div className="text-center mt-16">
                            <Text variant="body-2">
                                No pastes yet. Click &quot;New Paste&quot; in
                                the sidebar to create one.
                            </Text>
                        </div>
                    }
                    onItemClick={(paste) => router.push(`/${paste.code}`)}
                    filterItem={(filter) => (item) =>
                        item.preview.includes(filter)
                    }
                    renderItem={(paste) => {
                        const firstLine = paste.preview.split('\n')[0];
                        return (
                            <div className="py-2 px-4 cursor-pointer flex items-center justify-between gap-4 w-full">
                                <div className="flex items-center gap-4 flex-1 min-w-0">
                                    <Text variant="code-2" className="font-mono shrink-0">
                                        {paste.code}
                                    </Text>
                                    <Text variant="code-3" className="opacity-60 truncate flex-1">
                                        {firstLine}
                                    </Text>
                                </div>
                                <Text variant="caption-2" className="opacity-50 shrink-0 text-xs">
                                    {new Date(paste.createdAt).toLocaleString()}
                                </Text>
                            </div>
                        );
                    }}
                    itemHeight={36}
                />
            </div>
        </div>
    );
}
