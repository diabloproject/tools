"use client";
import { ClipboardButton } from "@gravity-ui/uikit";

export default function Copy({ text } : { text: string }) {
    return <ClipboardButton text={text} />
}
