"use client";
import { ThemeProvider } from "@gravity-ui/uikit";
import { AppLayout } from "./AppLayout";

export default function Wrapper({ children }: { children: React.ReactNode }) {
    return (
        <ThemeProvider theme="dark">
            <AppLayout>{children}</AppLayout>
        </ThemeProvider>
    );
}
