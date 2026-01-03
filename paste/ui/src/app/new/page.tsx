"use client";

import { TextArea, Button, Icon } from "@gravity-ui/uikit";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { PaperPlane, Pencil } from "@gravity-ui/icons";

export default function NewPastePage() {
    const [text, setText] = useState("");
    const [loading, setLoading] = useState(false);
    const router = useRouter();

    useEffect(() => {}, []);

    const handleCreate = async () => {
        if (!text.trim()) return;

        setLoading(true);
        try {
            const res = await fetch("/api/paste", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            if (!res.ok) {
                throw new Error("Failed to create paste");
            }

            const { code } = await res.json();
            router.push(`/${code}`);
        } catch (error) {
            console.error("Error creating paste:", error);
            alert("Failed to create paste");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-screen px-4 w-full overflow-scroll">
            <TextArea
                size="l"
                view="clear"
                value={text}
                onUpdate={setText}
                placeholder="Paste your text here..."
                autoFocus
                className="font-mono"
                controlProps={{
                    spellCheck: false,
                }}
            />
            <div className="absolute top-4 right-4">
                <Button onClick={handleCreate} view="action" loading={loading}>
                    <Icon data={PaperPlane} />
                    Submit
                </Button>
            </div>
        </div>
    );
}
