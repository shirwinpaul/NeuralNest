import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getLesson, LESSONS } from "@/data/lessons";
import { GLOSSARY } from "@/data/glossary";
import { LessonArt } from "@/components/academy/LessonArt";
import { Quiz } from "@/components/academy/Quiz";
import { ActivationsToy, GradientDescentToy } from "@/components/academy/Toys";

export function generateStaticParams() {
  return LESSONS.map((l) => ({ id: l.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const l = getLesson((await params).id);
  return { title: l ? l.title : "Lesson" };
}

export default async function LessonPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const lesson = getLesson(id);
  if (!lesson) notFound();
  const idx = LESSONS.findIndex((l) => l.id === id);
  const prev = LESSONS[idx - 1], next = LESSONS[idx + 1];
  const terms = GLOSSARY.filter((g) => g.lesson === id);

  return (
    <article className="mx-auto max-w-3xl space-y-6">
      <nav className="text-sm text-muted"><Link href="/academy" className="hover:text-cyan">← Academy</Link> · Lesson {idx + 1} of {LESSONS.length}</nav>
      <header className="flex flex-wrap items-center gap-4">
        <LessonArt kind={lesson.illustration} size={150} />
        <div>
          <h1 className="text-3xl font-bold">{lesson.emoji} {lesson.title}</h1>
          <p className="mt-1 text-lg text-muted">{lesson.summary}</p>
        </div>
      </header>

      {lesson.sections.map((s) => (
        <section key={s.heading} className="space-y-2">
          <h2 className="text-xl font-semibold text-cyan">{s.heading}</h2>
          {s.body.map((p, i) => <p key={i} className="leading-relaxed">{p}</p>)}
        </section>
      ))}

      {lesson.interactive === "gradient-descent" && <GradientDescentToy />}
      {lesson.interactive === "activations" && <ActivationsToy />}

      {lesson.tryIt && (
        <Link href={lesson.tryIt.href} className="btn btn-primary">🧪 Try it in the lab: {lesson.tryIt.label}</Link>
      )}

      {terms.length > 0 && (
        <aside className="panel p-4">
          <h2 className="mb-2 font-semibold">Key terms</h2>
          <dl className="grid gap-2 sm:grid-cols-2">
            {terms.map((t) => <div key={t.term}><dt className="font-medium">{t.term}</dt><dd className="text-sm text-muted">{t.def}</dd></div>)}
          </dl>
        </aside>
      )}

      <Quiz lessonId={lesson.id} questions={lesson.quiz} />

      <nav className="flex justify-between text-sm">
        {prev ? <Link href={`/academy/${prev.id}`} className="btn btn-sm">← {prev.title}</Link> : <span />}
        {next ? <Link href={`/academy/${next.id}`} className="btn btn-sm">{next.title} →</Link> : <Link href="/academy" className="btn btn-sm">Back to Academy</Link>}
      </nav>
    </article>
  );
}
