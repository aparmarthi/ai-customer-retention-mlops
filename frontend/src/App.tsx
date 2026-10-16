import { ApiProvider } from '@/context/ApiContext';
import { AssumptionsProvider } from '@/context/AssumptionsContext';
import { ToastProvider } from '@/context/ToastContext';
import { useHashRoute, type Route } from '@/hooks/useHashRoute';
import { AppLayout } from '@/components/layout/AppLayout';
import { OverviewPage } from '@/pages/OverviewPage';
import { SimulatorPage } from '@/pages/simulator/SimulatorPage';
import { EvidencePage } from '@/pages/evidence/EvidencePage';
import { ScorePage } from '@/pages/score/ScorePage';
import { MlopsPage } from '@/pages/MlopsPage';
import { ExperimentPage } from '@/pages/ExperimentPage';

const PAGES: Record<Route, () => JSX.Element> = {
  overview: OverviewPage,
  simulator: SimulatorPage,
  evidence: EvidencePage,
  score: ScorePage,
  mlops: MlopsPage,
  experiment: ExperimentPage,
};

function Shell() {
  const route = useHashRoute();
  const Page = PAGES[route];
  return (
    <AppLayout route={route}>
      <Page />
    </AppLayout>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <ApiProvider>
        <AssumptionsProvider>
          <Shell />
        </AssumptionsProvider>
      </ApiProvider>
    </ToastProvider>
  );
}
