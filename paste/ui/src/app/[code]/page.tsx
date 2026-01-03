import Copy from "@/components/copy";
import { ClipboardButton, Text } from "@gravity-ui/uikit";

async function fetchText(code: string) {
    const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';
    const res = await fetch(`${baseUrl}/api/paste?code=${code}`, {
        cache: 'no-store'
    });

    if (!res.ok) {
        throw new Error('Failed to fetch text');
    }

    const data = await res.json();
    return data.text;
}

export default async function PastePage({ params }: { params: Promise<{ code: string }> }) {
    const { code } = await params;
    const text = await fetchText(code);

    return (
        <div className="p-8 w-full h-screen relative">
            <div className="top-4 right-4 absolute">
                <Copy text={text} />
            </div>
            <Text variant="code-2" className="whitespace-pre">
                {text}
            </Text>
        </div>
    );
}
