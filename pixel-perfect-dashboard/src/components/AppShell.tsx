import { Link, useRouterState } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Bot, FileText, Home, Info, Map, Search, Settings, X,
} from "lucide-react";
import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import park from "@/assets/bengaluru-park.jpg";
import { fetchHealth } from "@/lib/api";

const nav = [
  ["Overview", Home, "/overview"],
  ["Explore", Search, "/explore"],
  ["Ward Intelligence", Map, "/explore"],
  ["HeatLens AI", Bot, "/heatlens"],
  ["Reports", FileText, "/reports"],
] as const;

interface SidebarState {
  open: boolean;
  toggle: () => void;
  close: () => void;
}

// Lives in __root.tsx, above both AppShell and every page component, so
// a page's own topbar button and AppShell's <aside> share the same state
// (a page calls useSidebarToggle() before it renders <AppShell>, so a
// context created inside AppShell would sit *below* the page in the tree
// and could never be read by it).
const SidebarContext = createContext<SidebarState>({
  open: false,
  toggle: () => {},
  close: () => {},
});

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  const value = useMemo<SidebarState>(
    () => ({
      open,
      toggle: () => setOpen((current) => !current),
      close: () => setOpen(false),
    }),
    [open],
  );

  return <SidebarContext.Provider value={value}>{children}</SidebarContext.Provider>;
}

export function useSidebarToggle() {
  return useContext(SidebarContext).toggle;
}

export function BackendStatus({ compact = false }: { compact?: boolean }) {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30000,
    retry: false,
  });

  const online = health.data?.status === "healthy";
  const label = health.isLoading
    ? "Checking backend…"
    : online
      ? "Backend online"
      : "Backend offline";

  return (
    <div className={compact ? "backend-status compact" : "backend-status"} title={label}>
      <i className={online ? "status-dot online" : "status-dot offline"} />
      {!compact && <span>{label}</span>}
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const { open: sidebarOpen, close: closeSidebar } = useContext(SidebarContext);
  const pathname = useRouterState({ select: (state) => state.location.pathname });

  const isActive = (path: string) =>
    path === "/overview" ? pathname === "/overview" : pathname.startsWith(path);

  return (
    <main className="app-shell">
      <aside className={sidebarOpen ? "sidebar sidebar-open" : "sidebar"}>
        <Link to="/" className="brand">
          <div className="brand-mark"><span /><i /></div>
          <div><b>HEATLENS</b><small>COOLER CITIES. BRIGHTER FUTURES.</small></div>
        </Link>
        <nav>
          {nav.map(([name, Icon, path]) => (
            <Link
              key={name}
              to={path}
              onClick={closeSidebar}
              className={name !== "Ward Intelligence" && isActive(path) ? "nav-item active" : "nav-item"}
            >
              <Icon /><span>{name}</span>
            </Link>
          ))}
        </nav>
        <div className="sidebar-lower">
          <div className="sidebar-photo">
            <img src={park} alt="Tree-lined park in Bengaluru" width={1024} height={768} loading="lazy" />
          </div>
          <BackendStatus />
          <div className="sidebar-links">
            <Link to="/settings" onClick={closeSidebar}><Settings /> Settings</Link>
            <Link to="/about" onClick={closeSidebar}><Info /> About</Link>
          </div>
          <blockquote>“Data for a cooler,<br /> healthier tomorrow.”</blockquote>
        </div>
      </aside>

      <section className="workspace">{children}</section>

      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          className="scrim"
          onClick={closeSidebar}
        >
          <X />
        </button>
      )}
    </main>
  );
}
