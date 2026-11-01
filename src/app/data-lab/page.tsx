import type { Metadata } from "next";
import { Suspense } from "react";
import { LabLoader } from "@/components/labs/LabLoader";

export const metadata: Metadata = { title: "Data Lab" };

export default function Page() {
  return (
    <Suspense>
      <LabLoader lab="data" />
    </Suspense>
  );
}
