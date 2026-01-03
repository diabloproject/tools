"use client";

import { AsideHeader } from "@gravity-ui/navigation";
import { Pencil } from "@gravity-ui/icons";
import { useState } from "react";
import { useRouter, usePathname } from "next/navigation";

interface AppLayoutProps {
    children: React.ReactNode;
}

export function AppLayout({ children }: AppLayoutProps) {
    const [compact, setCompact] = useState(true);
    const router = useRouter();
    const pathname = usePathname();

    return (
        <AsideHeader
            compact={compact}
            onChangeCompact={setCompact}
            logo={{
                text: "Paste",
                iconSrc: "/logo.png",
                iconSize: 42,
                onClick: () => router.push("/"),
            }}

            menuItems={[
                {
                    id: "new",
                    title: "New Paste",
                    type: "action",
                    icon: Pencil,
                    current: pathname === "/new",
                    onItemClick: () => router.push("/new"),
                },
            ]}
            renderContent={() => (
                <div className="h-full overflow-auto">
                    {children}
                </div>
            )}
            headerDecoration
        />
    );
}
