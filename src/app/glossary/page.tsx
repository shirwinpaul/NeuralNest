import type { Metadata } from "next";
import { GlossaryView } from "@/components/academy/GlossaryView";

export const metadata: Metadata = { title: "Glossary" };

export default function Page() {
  return <GlossaryView />;
}
