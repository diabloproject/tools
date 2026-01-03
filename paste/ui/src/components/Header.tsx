"use client";

import { Button, Icon } from "@gravity-ui/uikit";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { Text } from "@gravity-ui/uikit";
import { Pencil } from "@gravity-ui/icons"

export function Header() {
    const router = useRouter();

    return (
        <header className="px-8 py-4 border-b border-[var(--g-color-line-generic)] flex justify-between items-center">
            <div className="flex items-center">
                <Image
                    width={64}
                    height={64}
                    src="/favicon.ico"
                    alt="Logo"
                    className="relative -top-0.5"
                />
                <Text
                    variant="header-2"
                    className="text-2xl font-bold m-0 cursor-pointer"
                    onClick={() => router.push("/")}
                >
                    Paste
                </Text>
            </div>
            <Button view="action" onClick={() => router.push("/new")} size="l">
                <Icon data={Pencil} />
                New
            </Button>
        </header>
    );
}
