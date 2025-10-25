"use client";
import { ThemeProvider } from "@gravity-ui/uikit";
import type React from "react";

export default function DynamicRoot({
  children,
}: {
  children: React.ReactNode;
}) {
    return <ThemeProvider theme="dark">
        { children }
    </ThemeProvider>
}
