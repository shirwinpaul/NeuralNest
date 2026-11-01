import Link from "next/link";
import { Nesty } from "@/components/ui/Nesty";

export default function NotFound() {
  return (
    <div className="panel mx-auto max-w-md space-y-3 p-8 text-center">
      <Nesty size={80} mood="wow" className="mx-auto" />
      <h1 className="text-2xl font-bold">Page not found</h1>
      <p className="text-muted">Nesty searched every layer of the network and couldn&apos;t find that page.</p>
      <Link href="/" className="btn btn-primary">Back to the lab</Link>
    </div>
  );
}
