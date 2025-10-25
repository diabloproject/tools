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
                            <pre>
                                {(() => {
                                    // This is a simplified parser for ANSI color codes.
                                    // It handles basic foreground colors (30-37) and bright colors (90-97), plus reset (0).
                                    const ansiRegex = /(\u001b\[[0-9;]*[a-zA-Z])/;
                                    const parts = item.log.split(ansiRegex);
                                    const spans = [];
                                    let currentColor: string | undefined = undefined;

                                    const codeToColor = (code: string): string | undefined => {
                                        const codeNum = parseInt(code, 10);
                                        if (isNaN(codeNum)) return undefined;

                                        const colors = ['#000', '#A00', '#0A0', '#A50', '#00A', '#A0A', '#0AA', '#AAA'];
                                        const brightColors = ['#555', '#F55', '#5F5', '#FF5', '#55F', '#F5F', '#5FF', '#FFF'];

                                        if (codeNum >= 30 && codeNum <= 37) return colors[codeNum - 30];
                                        if (codeNum >= 90 && codeNum <= 97) return brightColors[codeNum - 90];
                                        return undefined;
                                    };

                                    for (let i = 0; i < parts.length; i++) {
                                        const part = parts[i];
                                        if (!part) continue;

                                        if (part.match(ansiRegex)) {
                                            const codes = part.substring(2, part.length - 1).split(';');
                                            for (const code of codes) {
                                                if (code === '0' || code === '') {
                                                    currentColor = undefined;
                                                } else {
                                                    const newColor = codeToColor(code);
                                                    if (newColor) {
                                                        currentColor = newColor;
                                                    }
                                                }
                                            }
                                        } else {
                                            spans.push(
                                                <span key={i} style={{ color: currentColor }}>
                                                    {part}
                                                </span>
                                            );
                                        }
                                    }

                                    return spans;
                                })()}
                            </pre>
                        </Text>
                    );
                }}
                itemHeight={(item) => 16}
                // itemsHeight={160}
                filterable={false}
            />
        </div>
    );
}
