"use client";

import { Modal, TextArea, Button } from "@gravity-ui/uikit";
import { useState } from "react";
import { useRouter } from "next/navigation";

interface NewPasteDialogProps {
    open: boolean;
    onClose: () => void;
}

export function NewPasteDialog({ open, onClose }: NewPasteDialogProps) {
    const [text, setText] = useState("");
    const [loading, setLoading] = useState(false);
    const router = useRouter();

    const handleCreate = async () => {
        if (!text.trim()) return;

        setLoading(true);
        try {
            const res = await fetch('/api/paste', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!res.ok) {
                throw new Error('Failed to create paste');
            }

            const { code } = await res.json();
            router.push(`/${code}`);
            onClose();
            setText("");
        } catch (error) {
            console.error('Error creating paste:', error);
            alert('Failed to create paste');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Modal open={open} onClose={onClose}>
            <div style={{ padding: '1.5rem', minWidth: '500px' }}>
                <h2 style={{ marginTop: 0 }}>New Paste</h2>
                <TextArea
                    value={text}
                    onUpdate={setText}
                    placeholder="Paste your text here..."
                    rows={15}
                    autoFocus
                    style={{ fontFamily: 'monospace', fontSize: '14px' }}
                />
                <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                    <Button onClick={onClose} disabled={loading}>
                        Cancel
                    </Button>
                    <Button view="action" onClick={handleCreate} disabled={loading || !text.trim()}>
                        {loading ? 'Creating...' : 'Create'}
                    </Button>
                </div>
            </div>
        </Modal>
    );
}
