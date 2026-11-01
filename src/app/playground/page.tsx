import type { Metadata } from "next";
import { Suspense } from "react";
import { LabLoader } from "@/components/labs/LabLoader";

export const metadata: Metadata = { title: "Neural Playground" };

export default function Page() {
  return (
    <Suspense>
      <LabLoader lab="playground" />
    </Suspense>
  );
}
