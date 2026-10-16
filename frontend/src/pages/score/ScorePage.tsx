import { useState } from 'react';
import { FileSpreadsheet, Terminal, User } from 'lucide-react';
import { useApi } from '@/context/ApiContext';
import { PageHeader } from '@/components/ui/primitives';
import { SingleScore } from '@/pages/score/SingleScore';
import { BatchScore } from '@/pages/score/BatchScore';
import { ApiConsole } from '@/pages/score/ApiConsole';

const TABS = [
  { id: 'single', label: 'Single customer', icon: User },
  { id: 'batch', label: 'Batch (CSV)', icon: FileSpreadsheet },
  { id: 'console', label: 'API console', icon: Terminal },
] as const;

type TabId = (typeof TABS)[number]['id'];

export function ScorePage() {
  const [tab, setTab] = useState<TabId>('single');
  const { isLive } = useApi();

  return (
    <div>
      <PageHeader
        eyebrow="Score Customers"
        title="Score customers against the live model"
        description={isLive ? 'Connected to the live scoring service. Every call shows its round-trip latency.' : 'Demo Mode: sample customers replay their recorded scores. Connect the API to score anything else.'}
      />
      <div role="tablist" aria-label="Scoring mode" className="mb-6 flex gap-1 overflow-x-auto border-b border-zinc-800">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            role="tab"
            id={`tab-${id}`}
            aria-selected={tab === id}
            aria-controls={`panel-${id}`}
            onClick={() => setTab(id)}
            className={`-mb-px flex shrink-0 items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
              tab === id ? 'border-emerald-400 text-zinc-50' : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="animate-fade-in">
        {tab === 'single' && <SingleScore />}
        {tab === 'batch' && <BatchScore />}
        {tab === 'console' && <ApiConsole />}
      </div>
    </div>
  );
}
