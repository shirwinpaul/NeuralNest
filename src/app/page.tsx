import Link from "next/link";
import { LABS } from "@/config/site";
import { HeroNetwork } from "@/components/ui/HeroNetwork";
import { Nesty } from "@/components/ui/Nesty";

export default function Home() {
  return (
    <div className="space-y-16">
      <section className="grid items-center gap-8 pt-6 md:grid-cols-2">
        <div className="space-y-5">
          <div className="flex flex-wrap gap-2">
            <span className="chip">🔒 Your data stays on your device</span>
            <span className="chip">⚡ Runs right in your browser — no installs</span>
          </div>
          <h1 className="text-4xl font-bold leading-tight sm:text-5xl">
            Don&apos;t just use AI —<br /><span className="bg-gradient-to-r from-cyan to-violet bg-clip-text text-transparent">build it.</span> 🧠⚡
          </h1>
          <p className="max-w-prose text-lg text-muted">
            NeuralNest is a science lab for real neural networks. Train them in your browser with TensorFlow.js,
            watch them learn live, test them, and find out <em>why</em> they behave the way they do.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/playground" className="btn btn-primary">🎯 Start in the Playground</Link>
            <Link href="/academy" className="btn">🎓 Learn the basics first</Link>
          </div>
        </div>
        <div className="panel glow p-4"><HeroNetwork /></div>
      </section>

      <section aria-labelledby="labs-h">
        <h2 id="labs-h" className="mb-4 text-2xl font-bold">Five labs to explore</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {LABS.map((l) => (
            <Link key={l.id} href={l.href} className="panel group flex flex-col gap-2 p-4 transition hover:glow">
              <span className="text-3xl" aria-hidden="true">{l.emoji}</span>
              <h3 className="font-semibold group-hover:text-cyan">{l.title}</h3>
              <p className="text-sm text-muted">{l.blurb}</p>
              {l.heavy && <span className="chip mt-auto w-fit">💻 best on a laptop</span>}
            </Link>
          ))}
        </div>
      </section>

      <section aria-labelledby="how-h" className="grid gap-6 md:grid-cols-[auto_1fr]">
        <Nesty size={120} mood="wow" className="mx-auto" />
        <div>
          <h2 id="how-h" className="mb-4 text-2xl font-bold">How it works</h2>
          <ol className="grid gap-4 sm:grid-cols-3">
            {[
              ["1. Collect data", "Pick a toy dataset, draw doodles, snap photos or load a CSV. It all stays in your browser."],
              ["2. Train for real", "Design a network, press Train and watch the loss drop — real maths on your device, nothing faked."],
              ["3. Test & understand", "Check honest metrics, find mistakes, and ask Nesty the tutor to explain your results."],
            ].map(([t, d]) => (
              <li key={t} className="panel p-4">
                <p className="font-semibold text-cyan">{t}</p>
                <p className="mt-1 text-sm text-muted">{d}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>
    </div>
  );
}
