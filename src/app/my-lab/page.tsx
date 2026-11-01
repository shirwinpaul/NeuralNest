import type { Metadata } from "next";
import { MyLab } from "@/components/mylab/MyLab";

export const metadata: Metadata = { title: "My Lab" };

export default function Page() {
  return <MyLab />;
}
