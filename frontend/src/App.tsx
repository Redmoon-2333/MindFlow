import { Suspense, lazy, useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import "./theme.css";
import { AUTH_REQUIRED_EVENT, bootstrapFromFragment, hasAuthenticatedSession } from "./api";
import { realtimeClient, requestNotificationPermission } from "./realtime";
import ErrorBoundary from "./components/ErrorBoundary";

// Route-level code splitting (architecture review 💡 19): heavy pages
// (ModelCenter, Diagnostics, Reports) load on demand to shrink the initial
// bundle.
const Login = lazy(() => import("./pages/Login"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Focus = lazy(() => import("./pages/Focus"));
const Activities = lazy(() => import("./pages/Activities"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Reports = lazy(() => import("./pages/Reports"));
const Intervention = lazy(() => import("./pages/Intervention"));
const Panel = lazy(() => import("./pages/Panel"));
const Chat = lazy(() => import("./pages/Chat"));
const Settings = lazy(() => import("./pages/Settings"));
const Diagnostics = lazy(() => import("./pages/Diagnostics"));
const ModelCenter = lazy(() => import("./pages/ModelCenter"));
const NotFound = lazy(() => import("./pages/NotFound"));

const NAV = [
  { to: "/", label: "仪表盘" },
  { to: "/focus", label: "专注分析" },
  { to: "/activities", label: "活动日志" },
  { to: "/analytics", label: "行为洞察" },
  { to: "/model-center", label: "模型中心" },
  { to: "/reports", label: "报告中心" },
  { to: "/intervention", label: "干预中心" },
  { to: "/panel", label: "专家面板" },
  { to: "/chat", label: "AI 对话" },
  { to: "/settings", label: "系统设置" },
  { to: "/diagnostics", label: "AI 诊断" },
];

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <aside className="sidebar">
        <h2>MindFlow</h2>
        <nav>
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}>
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">{children}</main>
    </>
  );
}

function AppRoutes() {
  return (
    <Suspense fallback={
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <div className="spinner" />
      </div>
    }>
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/focus" element={<Focus />} />
      <Route path="/activities" element={<Activities />} />
      <Route path="/analytics" element={<Analytics />} />
      <Route path="/reports" element={<Reports />} />
      <Route path="/intervention" element={<Intervention />} />
      <Route path="/panel" element={<Panel />} />
      <Route path="/chat" element={<Chat />} />
      <Route path="/settings" element={<Settings />} />
      <Route path="/diagnostics" element={<Diagnostics />} />
      <Route path="/model-center" element={<ModelCenter />} />
      <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
}

export default function App() {
  const [authenticated, setAuthenticated] = useState(hasAuthenticatedSession);
  const [bootstrapping, setBootstrapping] = useState(false);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);

  // Handle bootstrap ticket from URL hash on first load. The one-time ticket
  // is only removed from the URL after a successful exchange (see api.ts), so
  // a failed attempt stays retryable via the login page's retry button.
  useEffect(() => {
    if (authenticated) return;
    const params = new URLSearchParams(window.location.hash.slice(1));
    const ticket = params.get("bootstrap");
    if (!ticket) return;
    setBootstrapping(true);
    setBootstrapError(null);
    bootstrapFromFragment()
      .then((ok) => {
        if (ok) {
          setAuthenticated(true);
        }
      })
      .catch((error: unknown) => {
        setBootstrapError(
          error instanceof Error ? error.message : "认证失败，请重试",
        );
      })
      .finally(() => setBootstrapping(false));
  }, [authenticated]);

  useEffect(() => {
    const handleAuthRequired = () => setAuthenticated(false);
    window.addEventListener(AUTH_REQUIRED_EVENT, handleAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, handleAuthRequired);
  }, []);

    // Request browser notification permission for intervention alerts
  useEffect(() => {
    if (authenticated) requestNotificationPermission();
  }, [authenticated]);

  useEffect(() => {
    if (!authenticated) return;
    realtimeClient.connect();
    return () => realtimeClient.disconnect();
  }, [authenticated]);

  if (bootstrapping) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "var(--color-bg)" }}>
        <div className="card" style={{ width: 440, padding: 40, textAlign: "center" }}>
          <h1 style={{ fontSize: 28, color: "var(--color-primary)", marginBottom: 8 }}>MindFlow</h1>
          <p style={{ color: "var(--color-text-secondary)", marginBottom: 24 }}>认证中...</p>
          <div className="spinner" />
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <BrowserRouter>
        {!authenticated ? (
          <Login bootstrapError={bootstrapError} onClearBootstrapError={() => setBootstrapError(null)} />
        ) : (
          <Layout>
            <AppRoutes />
          </Layout>
        )}
      </BrowserRouter>
    </ErrorBoundary>
  );
}
