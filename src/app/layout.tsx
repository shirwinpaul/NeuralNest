import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { SettingsProvider } from "@/hooks/useSettings";
import { Header } from "@/components/ui/Header";
import { Footer } from "@/components/ui/Footer";
import { SettingsDialog } from "@/components/ui/SettingsDialog";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-space-grotesk", display: "swap" });
const sans = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", display: "swap" });

export const metadata: Metadata = {
  title: { default: "NeuralNest — Train-Your-Own-AI Lab", template: "%s · NeuralNest" },
  description: "Train real neural networks in your browser, watch them learn, and understand why. Your data stays on your device.",
};
export const viewport: Viewport = { themeColor: "#070b1a", width: "device-width", initialScale: 1 };

// Applied before paint so the saved theme doesn't flash.
const themeScript = `try{var s=JSON.parse(localStorage.getItem('nn.settings.v1')||'{}');if(s.theme==='light')document.documentElement.dataset.theme='light';if(s.reducedMotion==='on')document.documentElement.dataset.motion='off';if(s.reducedMotion==='off')document.documentElement.dataset.motion='on';}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" className={`${display.variable} ${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <SettingsProvider>
          <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 btn">Skip to content</a>
          <Header />
          <main id="main" className="mx-auto max-w-7xl px-4 py-6">{children}</main>
          <Footer />
          <SettingsDialog />
        </SettingsProvider>
      </body>
    </html>
  );
}
