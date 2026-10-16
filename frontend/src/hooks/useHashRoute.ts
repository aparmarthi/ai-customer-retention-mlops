import { useEffect, useState } from 'react';

export const ROUTES = ['overview', 'simulator', 'evidence', 'score', 'mlops', 'experiment'] as const;
export type Route = (typeof ROUTES)[number];

function parse(): Route {
  const r = window.location.hash.replace(/^#\/?/, '').split(/[/?]/)[0];
  return (ROUTES as readonly string[]).includes(r) ? (r as Route) : 'overview';
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(parse);
  useEffect(() => {
    if (!window.location.hash) window.history.replaceState(null, '', '#/overview');
    const onChange = () => {
      setRoute(parse());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}
