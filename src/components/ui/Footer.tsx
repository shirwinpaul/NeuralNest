import Link from "next/link";

export function Footer() {
  return (
    <footer className="no-print mt-16 border-t border-border">
      <div className="mx-auto grid max-w-7xl gap-4 px-4 py-8 text-sm text-muted md:grid-cols-3">
        <div>
          <p className="font-display text-base font-semibold text-text">Built by Anand Raj</p>
          <p className="mt-1">🔒 Your drawings, photos, webcam frames and CSV files never leave your device. The AI tutor only sees summary numbers.</p>
        </div>
        <div>
          <p className="font-semibold text-text">Credits</p>
          <ul className="mt-1 space-y-0.5">
            <li>MNIST digits — Yann LeCun, Corinna Cortes &amp; Chris Burges (CC BY-SA 3.0)</li>
            <li>Iris dataset — R. A. Fisher (1936), UCI ML Repository</li>
            <li>MobileNet — Google (Apache-2.0)</li>
            <li>TensorFlow.js — Google (Apache-2.0)</li>
          </ul>
        </div>
        <div>
          <p className="font-semibold text-text">Learn</p>
          <ul className="mt-1 space-y-0.5">
            <li><Link href="/academy" className="hover:text-cyan">ML Academy</Link></li>
            <li><Link href="/glossary" className="hover:text-cyan">Glossary</Link></li>
            <li><Link href="/my-lab" className="hover:text-cyan">My Lab</Link></li>
          </ul>
        </div>
      </div>
    </footer>
  );
}
