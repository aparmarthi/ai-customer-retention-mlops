import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';
import type { Assumptions } from '@/lib/economics';
import { DEFAULT_SCENARIO, clampTo, decodeScenario, hashQuery, type ScenarioState } from '@/lib/scenario';

interface AssumptionsContextValue extends Assumptions {
  subscribers: number;
  k: number;
  setValuePerSave: (v: number) => void;
  setCostPerContact: (v: number) => void;
  setSaveRate: (v: number) => void;
  setSubscribers: (v: number) => void;
  setK: (v: number) => void;
  reset: () => void;
}

const AssumptionsContext = createContext<AssumptionsContextValue | null>(null);

export function AssumptionsProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ScenarioState>(() => decodeScenario(hashQuery(window.location.hash)));

  const value = useMemo<AssumptionsContextValue>(() => {
    const set = (key: keyof ScenarioState) => (v: number) => setState((s) => ({ ...s, [key]: clampTo(key, v) }));
    return {
      ...state,
      setValuePerSave: set('valuePerSave'),
      setCostPerContact: set('costPerContact'),
      setSaveRate: set('saveRate'),
      setSubscribers: set('subscribers'),
      setK: set('k'),
      reset: () => setState(DEFAULT_SCENARIO),
    };
  }, [state]);

  return <AssumptionsContext.Provider value={value}>{children}</AssumptionsContext.Provider>;
}

export function useAssumptions(): AssumptionsContextValue {
  const ctx = useContext(AssumptionsContext);
  if (!ctx) throw new Error('useAssumptions must be used inside AssumptionsProvider');
  return ctx;
}
