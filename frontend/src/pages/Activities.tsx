import { useState, useEffect, useCallback } from "react";
import { getActivities, getCurrentActivity } from "../api";

const PAGE_SIZE = 20;

export default function Activities() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [currentActivity, setCurrentActivity] = useState<any>(null);
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [search, setSearch] = useState("");

  const fetchCurrent = useCallback(async () => {
    try {
      const ca = await getCurrentActivity();
      setCurrentActivity(ca);
    } catch {
      setCurrentActivity(null);
    }
  }, []);

  const fetchActivities = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, string> = {
        page: String(page),
        page_size: String(PAGE_SIZE),
      };
      if (startDate) params.start = startDate;
      if (endDate) params.end = endDate;
      if (search) params.q = search;
      const result = await getActivities(params);
      setItems(result.items ?? []);
      setTotal(result.total ?? 0);
    } catch (e: any) {
      setError(e.message ?? "加载失败");
    } finally {
      setLoading(false);
    }
  }, [page, startDate, endDate, search]);

  useEffect(() => {
    fetchCurrent();
  }, [fetchCurrent]);

  useEffect(() => {
    fetchActivities();
  }, [fetchActivities]);

  useEffect(() => {
    setPage(1);
  }, [startDate, endDate, search]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const formatDuration = (seconds: number): string => {
    if (seconds == null) return "--";
    const s = Math.round(seconds);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  };

  const classificationBadge = (cls: string) => {
    const map: Record<string, string> = {
      productive: "badge-success",
      neutral: "badge-info",
      distracting: "badge-warning",
      unclassified: "",
    };
    return map[cls] ?? "";
  };

  const statusBadge = (sts: string) => {
    const map: Record<string, string> = {
      active: "badge-success",
      idle: "badge-info",
      away: "badge-warning",
      offline: "badge-danger",
    };
    return map[sts] ?? "";
  };

  const formatTime = (ts: string): string => {
    if (!ts) return "--";
    try {
      const d = new Date(ts);
      return d.toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return ts;
    }
  };

  return (
    <div>
      <div className="header">
        <h1>活动日志</h1>
        <p>追踪与回放应用使用行为</p>
      </div>

      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-sm mt8" onClick={fetchActivities} style={{ marginLeft: 12 }}>
            重试
          </button>
        </div>
      )}

      {/* Current Activity */}
      <div className="card mb24">
        <h3>当前活动</h3>
        {currentActivity ? (
          <div className="flex gap8" style={{ alignItems: "center" }}>
            <span style={{ fontWeight: 600 }}>{currentActivity.app_name ?? "--"}</span>
            {currentActivity.window_title && (
              <span style={{ color: "var(--color-text-secondary)" }}>
                — {currentActivity.window_title}
              </span>
            )}
            {currentActivity.classification && (
              <span className={`badge ${classificationBadge(currentActivity.classification)}`}>
                {currentActivity.classification}
              </span>
            )}
          </div>
        ) : (
          <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
            暂无活跃窗口
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="card mb24">
        <div className="flex gap16 mb16" style={{ flexWrap: "wrap", alignItems: "flex-end" }}>
          <div>
            <label style={{ fontSize: 12, color: "var(--color-text-tertiary)", display: "block", marginBottom: 4 }}>
              开始日期
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ width: 160 }}
            />
          </div>
          <div>
            <label style={{ fontSize: 12, color: "var(--color-text-tertiary)", display: "block", marginBottom: 4 }}>
              结束日期
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ width: 160 }}
            />
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: 12, color: "var(--color-text-tertiary)", display: "block", marginBottom: 4 }}>
              搜索
            </label>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="按应用名、窗口标题搜索..."
            />
          </div>
        </div>
      </div>

      {/* Data Table */}
      <div className="card mb24">
        {loading ? (
          <div className="spinner" />
        ) : items.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--color-text-tertiary)", textAlign: "center", padding: 40 }}>
            暂无活动记录
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>应用名称</th>
                  <th>窗口标题</th>
                  <th>进程</th>
                  <th>时长</th>
                  <th>分类</th>
                  <th>状态</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item: any, idx: number) => (
                  <tr key={item.id ?? idx}>
                    <td style={{ whiteSpace: "nowrap" }}>
                      {formatTime(item.time ?? item.timestamp ?? item.created_at)}
                    </td>
                    <td style={{ fontWeight: 500 }}>{item.app_name ?? "--"}</td>
                    <td style={{ maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {item.window_title ?? "--"}
                    </td>
                    <td>{item.process ?? "--"}</td>
                    <td>{formatDuration(item.duration_seconds ?? item.duration)}</td>
                    <td>
                      {item.classification ? (
                        <span className={`badge ${classificationBadge(item.classification)}`}>
                          {item.classification}
                        </span>
                      ) : (
                        "--"
                      )}
                    </td>
                    <td>
                      {item.status ? (
                        <span className={`badge ${statusBadge(item.status)}`}>
                          {item.status}
                        </span>
                      ) : (
                        "--"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {!loading && total > 0 && (
          <div className="flex-between mt16" style={{ marginTop: 16 }}>
            <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              共 {total} 条，第 {page} / {totalPages} 页
            </span>
            <div className="flex gap8">
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                上一页
              </button>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                下一页
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
