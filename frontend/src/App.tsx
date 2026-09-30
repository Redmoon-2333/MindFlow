import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import "./theme.css";
import "./shell.css";
import "./assets/reference-font/iconfont.css";
import {
  AUTH_REQUIRED_EVENT,
  bootstrapFromFragment,
  hasAuthenticatedSession,
  logout,
  getErrorMessage,
} from "./api";
import logo from "./assets/logo.png";
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
const Execution = lazy(() => import("./pages/Execution"));
const Panel = lazy(() => import("./pages/Panel"));
const Chat = lazy(() => import("./pages/Chat"));
const Settings = lazy(() => import("./pages/Settings"));
const Diagnostics = lazy(() => import("./pages/Diagnostics"));
const ModelCenter = lazy(() => import("./pages/ModelCenter"));
const NotFound = lazy(() => import("./pages/NotFound"));

/** The nine navigation entries of the reference build, in its original order. */
const NAV = [
  { to: "/", label: "仪表盘", icon: "icon-yibiaopan", end: true },
  { to: "/focus", label: "专注分析", icon: "icon-yanjing" },
  { to: "/activities", label: "活动日志", icon: "icon-huodong" },
  { to: "/analytics", label: "行为洞察", icon: "icon-shoushidongzuo_dianji_click-tap-two" },
  { to: "/reports", label: "报告中心", icon: "icon-dubanbaogao" },
  { to: "/intervention", label: "干预中心", icon: "icon-ganyufangan" },
  { to: "/panel", label: "专家面板", icon: "icon-zhuanjia" },
  { to: "/chat", label: "AI 对话", icon: "icon-wuguan" },
  { to: "/settings", label: "系统设置", icon: "icon-shezhi" },
];

/** Local-only capabilities the reference does not ship; parked behind the
 *  advanced entry so the nine reference items stay untouched. */
const ADVANCED_NAV = [
  { to: "/model-center", label: "模型中心", icon: "icon-shujuku" },
  { to: "/execution", label: "干预执行", icon: "icon-tixing" },
  { to: "/diagnostics", label: "AI 诊断", icon: "icon-bodong" },
];

const ADVANCED_PATHS = new Set(ADVANCED_NAV.map((item) => item.to));

function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  // The breakpoint must be state, not read inside the click handler: the
  // button's label/aria-expanded are rendered from it, and the reference CSS
  // switches to the drawer at max-width 1023px (shell.css).
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 1023);
  const [groupOpen, setGroupOpen] = useState(() => ADVANCED_PATHS.has(window.location.pathname));
  const [logoutError, setLogoutError] = useState<string | null>(null);
  const [loggingOut, setLoggingOut] = useState(false);
  const [confirmLogout, setConfirmLogout] = useState(false);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  /** Element focus returns to when the dialog closes (WCAG 2.4.3). */
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const navToggleRef = useRef<HTMLButtonElement | null>(null);

  const title = useMemo(() => {
    const all = [...NAV, ...ADVANCED_NAV];
    return all.find((item) => item.to === location.pathname)?.label ?? "页面未找到";
  }, [location.pathname]);

  // Entering an advanced route auto-expands the advanced group so the active
  // item is always visible, per the agreed navigation rule.
  useEffect(() => {
    if (ADVANCED_PATHS.has(location.pathname)) setGroupOpen(true);
    setDrawerOpen(false);
    document.title = `${title} | MindFlow`;
  }, [location.pathname, title]);

  useEffect(() => {
    const breakpoint = window.matchMedia("(max-width: 1023px)");
    const onChange = () => {
      setIsMobile(breakpoint.matches);
      setDrawerOpen(false);
    };
    breakpoint.addEventListener("change", onChange);
    return () => breakpoint.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!isMobile || !drawerOpen || confirmLogout) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      setDrawerOpen(false);
      navToggleRef.current?.focus();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isMobile, drawerOpen, confirmLogout]);

  useEffect(() => {
    if (!confirmLogout) return;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setConfirmLogout(false);
        // Focus went to <body> when the dialog unmounted; hand it back to
        // the control that opened it so keyboard users are not stranded.
        window.setTimeout(() => returnFocusRef.current?.focus(), 0);
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      // Keep Tab inside the dialog: the reference modal traps focus.
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [confirmLogout]);

  /** Close the dialog and restore focus to its trigger. */
  const closeLogout = () => {
    setConfirmLogout(false);
    window.setTimeout(() => returnFocusRef.current?.focus(), 0);
  };

  const openLogout = (event: React.MouseEvent<HTMLButtonElement>) => {
    returnFocusRef.current = event.currentTarget;
    setConfirmLogout(true);
  };

  const handleLogout = async () => {
    setLoggingOut(true);
    setLogoutError(null);
    try {
      await logout();
      setConfirmLogout(false);
    } catch (error) {
      setLogoutError(getErrorMessage(error, "退出失败，请重试"));
    } finally {
      setLoggingOut(false);
    }
  };

  /** Is the sidebar on screen right now? Desktop and mobile answer this from
   *  different state, but there is exactly one truth to render. */
  const navVisible = isMobile ? drawerOpen : !collapsed;
  /** Desktop-only "sidebar pushed off-canvas"; the ≤1023px CSS hides it anyway. */
  const wide = !isMobile && collapsed;
  const closeNav = () => {
    if (isMobile) setDrawerOpen(false);
    else setCollapsed(true);
  };

  return (
    <div className="mf-body">
      <div className="mf-ambient-glow" aria-hidden="true" />
      <a href="#main-content" className="mf-skip-link">
        跳转到内容
      </a>

      {isMobile && navVisible && (
        <button className="mf-scrim" aria-label="关闭导航" onClick={closeNav} />
      )}

      <aside
        className={`mf-navbar${wide ? " mf-navbar--collapsed" : ""}${drawerOpen ? " mf-navbar--open" : ""}${confirmLogout ? " mf-navbar--blur" : ""}`}
        aria-label="主导航"
        // Off-canvas nav must leave the tab order (audit F4 keyboard check).
        inert={!navVisible}
      >
        <NavLink to="/" className="mf-logo" aria-label="MindFlow 首页">
          <img className="mf-logo-img" src={logo} alt="" width={40} height={40} />
          <span className="mf-logo-title">MindFlow</span>
        </NavLink>
        <div className="mf-logo-line" aria-hidden="true" />

        <nav className="mf-nav-items" aria-label="主导航">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end ?? false}
              className={({ isActive }) => `mf-nav-item${isActive ? " mf-nav-item--active" : ""}`}
            >
              <span className={`iconfont ${n.icon}`} aria-hidden="true" />
              {n.label}
            </NavLink>
          ))}

          <div className="mf-nav-group">
            <button
              type="button"
              className="mf-nav-group-toggle"
              aria-expanded={groupOpen}
              aria-controls="advanced-nav"
              onClick={() => setGroupOpen((open) => !open)}
            >
              <span>高级功能</span>
              <span
                className={`iconfont icon-double-arrow-left-full mf-nav-group-chevron${groupOpen ? " mf-nav-group-chevron--open" : ""}`}
                aria-hidden="true"
              />
            </button>
            {groupOpen && (
              <div className="mf-nav-group-list" id="advanced-nav">
                {ADVANCED_NAV.map((n) => (
                  <NavLink
                    key={n.to}
                    to={n.to}
                    className={({ isActive }) => `mf-nav-item${isActive ? " mf-nav-item--active" : ""}`}
                  >
                    <span className={`iconfont ${n.icon}`} aria-hidden="true" />
                    {n.label}
                  </NavLink>
                ))}
              </div>
            )}
          </div>
        </nav>

        <button type="button" className="mf-user" onClick={openLogout} aria-haspopup="dialog">
          <span className="mf-user-name">本地会话</span>
          <span className="iconfont icon-tuichu mf-user-icon" aria-hidden="true" />
          <span className="mf-skip-link">退出登录</span>
        </button>
      </aside>

      <div
        className={`mf-content${wide ? " mf-content--wide" : ""}${confirmLogout ? " mf-navbar--blur" : ""}`}
      >
        <header className="mf-header">
          <button
            type="button"
            className="mf-header-toggle"
            ref={navToggleRef}
            aria-label={navVisible ? "收起导航" : "展开导航"}
            aria-expanded={navVisible}
            onClick={() => {
              if (isMobile) setDrawerOpen((open) => !open);
              else setCollapsed((value) => !value);
            }}
          >
            <span className="iconfont icon-icon_cebianlan" aria-hidden="true" />
          </button>
          <span className="mf-header-name">{title}</span>
        </header>

        <main className="mf-main" id="main-content" tabIndex={-1} key={location.pathname}>
          {children}
        </main>
      </div>

      {confirmLogout && (
        <div className="mf-modal-overlay" onMouseDown={closeLogout}>
          <div
            className="mf-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="logout-title"
            aria-describedby="logout-desc"
            ref={dialogRef}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="mf-dialog-close"
              aria-label="关闭"
              onClick={closeLogout}
              disabled={loggingOut}
            >
              <span className="iconfont icon-chacha" aria-hidden="true" />
            </button>
            <h2 className="mf-dialog-title" id="logout-title">
              确认退出当前账号吗？
            </h2>
            <p className="mf-dialog-desc" id="logout-desc">
              退出后需要重新进行本地认证。
            </p>
            {logoutError && (
              <div className="mf-dialog-error" role="alert">
                {logoutError}
              </div>
            )}
            <button
              type="button"
              className="mf-dialog-yes"
              onClick={handleLogout}
              disabled={loggingOut}
            >
              {loggingOut ? "退出中..." : "确认退出"}
            </button>
            <button
              type="button"
              className="mf-dialog-no"
              ref={closeRef}
              onClick={closeLogout}
              disabled={loggingOut}
            >
              取消
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function AppRoutes() {
  return (
    <Suspense
      fallback={
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "60vh",
          }}
        >
          <div className="spinner" />
        </div>
      }
    >
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/focus" element={<Focus />} />
        <Route path="/activities" element={<Activities />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/intervention" element={<Intervention />} />
        <Route path="/execution" element={<Execution />} />
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
        setBootstrapError(error instanceof Error ? error.message : "认证失败，请重试");
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
      <div className="mf-login-page">
        <div className="login-card">
          <h1 className="login-title">MindFlow</h1>
          <p className="login-subtitle">认证中...</p>
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
