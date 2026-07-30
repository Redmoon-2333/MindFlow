import { NavLink } from "react-router-dom";
import "./model-center.css";

export default function NotFound() {
  return (
    <div className="not-found-page">
      <h1 className="nf-title">404</h1>
      <p className="nf-desc">页面未找到</p>
      <NavLink to="/" className="btn nf-link">
        返回仪表盘
      </NavLink>
    </div>
  );
}
