import type { Metadata } from "next";
import { AcademyIndex } from "@/components/academy/AcademyIndex";

export const metadata: Metadata = { title: "ML Academy" };

export default function Page() {
  return <AcademyIndex />;
}
