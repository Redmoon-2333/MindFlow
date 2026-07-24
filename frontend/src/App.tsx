import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import "./theme.css";
import { getTokenValue } from "./api";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Focus from "./pages/Focus";
import Activities from "./pages/Activities";
import Analytics from "./pages/Analytics";
import Reports from "./pages/Reports";
import Intervention from "./pages/Intervention";
import Panel from "./pages/Panel";
import Chat from "./pages/Chat";
import Settings from "./pages/Settings";

const NAV = [
  { to: "/", label: "仪表盘" },
  { to: "/focus", label: "专注分析" },
  { to: "/activities", label: "活动日志" },
  { to: "/analytics", label: "行为洞察" },
  { to: "/reports", label: "报告中心" },
  { to: "/intervention", label: "干预中心" },
  { to: "/panel", label: "专家面板" },
  { to: "/chat", label: "AI 对话" },
  { to: "/settings", label: "系统设置" },
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
    </Routes>
  );
}

export default function App() {
  const token = getTokenValue();

  return (
    <BrowserRouter>
      {!token ? (
        <Login />
      ) : (
        <Layout>
          <AppRoutes />
        </Layout>
      )}
    </BrowserRouter>
  );
}
