import { useState, type ReactNode } from 'react';
import {
  Activity,
  BarChart3,
  Beaker,
  Database,
  Github,
  LayoutDashboard,
  Linkedin,
  Menu,
  MonitorPlay,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  X,
} from 'lucide-react';
import type { Route } from '@/hooks/useHashRoute';
import { StatusPill } from '@/components/layout/StatusPill';
import { SubscriberInput } from '@/components/layout/SubscriberInput';
import { SettingsDrawer } from '@/components/layout/SettingsDrawer';
import { CaveatChip } from '@/components/ui/CaveatChip';

const NAV: Array<{ route: Route; label: string; icon: typeof LayoutDashboard; hint: string }> = [
  { route: 'overview', label: 'Overview', icon: LayoutDashboard, hint: 'Executive one-pager' },
  { route: 'simulator', label: 'Policy Simulator', icon: SlidersHorizontal, hint: 'Who to contact, and what it nets' },
  { route: 'evidence', label: 'Model Evidence', icon: BarChart3, hint: 'Validation and benchmark' },
  { route: 'score', label: 'Score Customers', icon: Target, hint: 'Live API' },
  { route: 'mlops', label: 'MLOps & Monitoring', icon: Activity, hint: 'Deploy, monitor, retrain' },
  { route: 'experiment', label: 'Experiment Plan', icon: Beaker, hint: 'Simulated to measured ROI' },
];

const LINKS = [
  { href: 'https://github.com/aparmarthi/ai-customer-retention-mlops', label: 'GitHub', icon: Github },
  { href: 'https://amey-churn-predictor.streamlit.app/', label: 'Original Streamlit app', icon: MonitorPlay },
  { href: 'https://www.linkedin.com/in/ameyparmarthi/', label: 'LinkedIn', icon: Linkedin },
  { href: 'https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge', label: 'Dataset', icon: Database },
];

function Brand() {
  return (
    <a href="#/overview" className="flex items-center gap-2.5">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 shadow-lg shadow-emerald-500/20">
        <ShieldCheck className="h-[18px] w-[18px] text-zinc-950" strokeWidth={2.5} aria-hidden />
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-semibold text-zinc-50">Churn Decision</span>
        <span className="block text-[11px] font-medium text-zinc-500">Intelligence</span>
      </span>
    </a>
  );
}

function NavList({ route, onNavigate }: { route: Route; onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary" className="flex flex-col gap-1">
      {NAV.map(({ route: r, label, icon: Icon, hint }) => {
        const active = r === route;
        return (
          <a
            key={r}
            href={`#/${r}`}
            onClick={onNavigate}
            aria-current={active ? 'page' : undefined}
            className={`group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
              active ? 'bg-zinc-800/80 text-zinc-50' : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100'
            }`}
          >
            {active && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-emerald-400" />}
            <Icon className={`h-4 w-4 shrink-0 ${active ? 'text-emerald-400' : 'text-zinc-500 group-hover:text-zinc-300'}`} aria-hidden />
            <span className="min-w-0">
              <span className="block truncate font-medium">{label}</span>
              <span className="block truncate text-[11px] text-zinc-500">{hint}</span>
            </span>
          </a>
        );
      })}
    </nav>
  );
}

export function AppLayout({ route, children }: { route: Route; children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <div className="min-h-screen bg-zinc-950 bg-[radial-gradient(ellipse_at_top,rgba(16,185,129,0.06),transparent_50%)]">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-zinc-800/80 bg-zinc-950/90 px-4 py-5 backdrop-blur lg:flex">
        <Brand />
        <div className="mt-8 flex-1">
          <NavList route={route} />
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-[11px] text-zinc-500">
          KKBox WSDM 2018 · LightGBM champion · Cloud Run
        </div>
      </aside>

      {menuOpen && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 animate-fade-in bg-black/60" onClick={() => setMenuOpen(false)} />
          <aside className="absolute inset-y-0 left-0 flex w-72 flex-col border-r border-zinc-800 bg-zinc-950 px-4 py-5 shadow-2xl">
            <div className="flex items-center justify-between">
              <Brand />
              <button onClick={() => setMenuOpen(false)} className="btn-ghost p-2" aria-label="Close navigation">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-8">
              <NavList route={route} onNavigate={() => setMenuOpen(false)} />
            </div>
            <div className="mt-6 border-t border-zinc-800 pt-4">
              <SubscriberInput id="subscriber-base-mobile" />
            </div>
          </aside>
        </div>
      )}

      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 border-b border-zinc-800/80 bg-zinc-950/85 backdrop-blur-md">
          <div className="flex h-14 items-center gap-3 px-4 md:px-8">
            <button onClick={() => setMenuOpen(true)} className="btn-ghost p-2 lg:hidden" aria-label="Open navigation">
              <Menu className="h-5 w-5" />
            </button>
            <div className="min-w-0 flex-1">
              <StatusPill />
            </div>
            <div className="hidden md:block">
              <SubscriberInput />
            </div>
            <div className="hidden h-6 w-px bg-zinc-800 md:block" />
            <div className="flex items-center">
              {LINKS.map(({ href, label, icon: Icon }) => (
                <a
                  key={href}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={label}
                  aria-label={label}
                  className={`btn-ghost p-2 ${label === 'Original Streamlit app' || label === 'Dataset' ? 'hidden sm:inline-flex' : ''}`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden text-xs 2xl:inline">{label}</span>
                </a>
              ))}
              <button onClick={() => setSettingsOpen(true)} className="btn-ghost p-2" aria-label="Open settings">
                <Settings className="h-4 w-4" />
              </button>
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-7xl px-4 py-8 md:px-8 md:py-10">
          <div key={route} className="animate-slide-up">
            {children}
          </div>
          <footer className="mt-12 flex flex-col gap-3 border-t border-zinc-800/80 pt-6 sm:flex-row sm:items-center sm:justify-between">
            <CaveatChip />
            <p className="text-[11px] text-zinc-600">Churn Decision Intelligence · KKBox WSDM 2018</p>
          </footer>
        </main>
      </div>

      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
