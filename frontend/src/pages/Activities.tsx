import NativeDateField from "../components/NativeDateField";
import { useState, useEffect, useCallback, useRef } from "react";
import { getActivities, getCurrentActivity, getErrorMessage } from "../api";
import type { ActivityItem } from "../api";
import Surface from "../components/signal/Surface";
import HelpTip from "../components/signal/HelpTip";
import "./activities.css";

/** Reference page size — the original table shows 10 rows per page. */
const PAGE_SIZE = 10;
/** Debounce for the server-side text search (plan: 300ms). */
const SEARCH_DEBOUNCE_MS = 300;

export default function Activities() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [currentActivity, setCurrentActivity] = useState<ActivityItem | null>(null);
  const [items, setItems] = useState<ActivityItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [search, setSearch] = useState("");
  // Debounced mirror of `search`; the request only ever sees this value so
  // keystrokes do not each trigger a round trip.
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [isRawDebugVisible, setIsRawDebugVisible] = useState(false);
  const [columns, setColumns] = useState({
    app: true,
    title: true,
    process: true,
    duration: true,
    status: true,
    raw: false,
  });

  const searchTimer = useRef<number | null>(null);
  // Request-sequence guard: a late response for an older filter must not
  // overwrite the current page (stale-overwrite race).
  const requestSeqRef = useRef(0);

  const fetchCurrent = useCallback(async () => {
    try {
      const ca = await getCurrentActivity();
      setCurrentActivity(ca);
    } catch {
      setCurrentActivity(null);
    }
  }, []);

  const fetchActivities = useCallback(async () => {
    const seq = ++requestSeqRef.current;
    setLoading(true);
    setError(null);
    try {
      const result = await getActivities({
        page,
        page_size: PAGE_SIZE,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        q: debouncedSearch.trim() || undefined,
      });
      if (seq !== requestSeqRef.current) return; // stale response
      setItems(result.items ?? []);
      // Server-side search returns the filtered total, so pagination and the
      // row list always describe the same result set.
      setTotal(result.total ?? 0);
    } catch (e: unknown) {
      if (seq !== requestSeqRef.current) return;
      setError(getErrorMessage(e, "加载失败"));
      setItems([]);
      setTotal(0);
    } finally {
      if (seq === requestSeqRef.current) setLoading(false);
    }
  }, [page, startDate, endDate, debouncedSearch]);

  useEffect(() => {
    fetchCurrent();
  }, [fetchCurrent]);

  useEffect(() => {
    fetchActivities();
  }, [fetchActivities]);

  // Debounce the search box, then reset to page 1 so a new filter never shows
  // an out-of-range page.
  useEffect(() => {
    if (searchTimer.current != null) window.clearTimeout(searchTimer.current);
    searchTimer.current = window.setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      if (searchTimer.current != null) window.clearTimeout(searchTimer.current);
    };
  }, [search]);

  useEffect(() => {
    setPage(1);
  }, [startDate, endDate]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const formatDuration = (seconds: number): string => {
    if (seconds == null) return "—";
    const s = Math.round(seconds);
    if (s < 60) return `${s} 秒`;
    return `${Math.round(s / 60)} 分钟`;
  };

  const formatTime = (ts: string): string => {
    if (!ts) return "—";
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

  const columnToggles: Array<{ key: keyof typeof columns; label: string }> = [
    { key: "app", label: "应用名称" },
    { key: "title", label: "窗口标题" },
    { key: "process", label: "进程" },
    { key: "duration", label: "时长" },
    { key: "status", label: "状态" },
    { key: "raw", label: "原始字段" },
  ];

  return (
    <>
      <div className="activities">
        <p className="page-lede">按时间查看记录到的应用和窗口，可按日期筛选或搜索。</p>

        <Surface as="section" className="present-acti">
          <h2 className="sh-title present-acti-title">当前活动</h2>
          <div className="present-acti-line">
            {currentActivity ? (
              <>
                <b>{currentActivity.data?.app_name || "未知应用"}</b>
                {" — "}
                <span className="present-acti-window">
                  {currentActivity.data?.window_title || currentActivity.data?.process_name || "暂无活跃窗口"}
                </span>
              </>
            ) : (
              <span className="present-acti-window">暂无活跃窗口</span>
            )}
          </div>
        </Surface>

        <div className="page-toolbar choose-time">
          <div className="start-time-box">
            <div className="act-datepicker mf-picker">
              <NativeDateField label="开始日期" value={startDate} onChange={setStartDate} allowEmpty />
            </div>
          </div>

          <div className="end-time-box">
            <div className="act-datepicker mf-picker">
              <NativeDateField label="结束日期" value={endDate} onChange={setEndDate} allowEmpty />
            </div>
          </div>

          <div className="search-box">
            <div className="text">搜索</div>
            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="按应用名/窗口名搜索…"
              aria-label="搜索活动"
            />
          </div>
        </div>

        <div className="column-options">
          <input
            type="checkbox"
            id="act-raw-toggle"
            checked={isRawDebugVisible}
            onChange={(event) => {
              setIsRawDebugVisible(event.target.checked);
              setColumns((current) => ({ ...current, raw: event.target.checked }));
            }}
          />
          {/* §8.21.6: the tip belongs to the label text, not to the flex row —
              as a sibling the row's 16px gap pushed it off its word. */}
          <label className="column-toggle" htmlFor="act-raw-toggle">
            显示事件编号
            <HelpTip
              term="事件编号"
              what="多显示一列事件类型和编号，方便排查问题。"
              effect="只影响显示，不会改动数据。"
            />
          </label>
          {/* Local addition: the plan requires the raw-field switch to
              really control column display, so the remaining column
              switches sit with it rather than inside the table block. */}
          {columnToggles
            .filter((toggle) => toggle.key !== "raw")
            .map((toggle) => (
              <label key={toggle.key} className="column-toggle">
                <input
                  type="checkbox"
                  checked={columns[toggle.key]}
                  onChange={(event) =>
                    setColumns((current) => ({ ...current, [toggle.key]: event.target.checked }))
                  }
                />
                {toggle.label}
              </label>
            ))}
        </div>

        {error && (
          <div className="error-box" role="alert">
            {error}
            <button className="btn btn-secondary btn-sm error-retry" onClick={fetchActivities}>
              重试
            </button>
          </div>
        )}

        <Surface as="section" className="detail-list">
          {loading ? (
            <div className="spinner" />
          ) : items.length === 0 ? (
            <div className="table-empty">
              {debouncedSearch.trim() ? "当前搜索无匹配记录" : "暂无活动记录"}
            </div>
          ) : (
            <div className="table-scroll">
              <table className="simple-table">
                <thead>
                  <tr>
                    <th>时间</th>
                    {columns.app && <th>应用名称</th>}
                    {columns.title && <th>窗口标题</th>}
                    {columns.process && <th>进程</th>}
                    {columns.duration && <th>时长</th>}
                    {columns.status && <th>状态</th>}
                    {columns.raw && <th>原始字段</th>}
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, idx) => (
                    <tr key={item.id ?? idx}>
                      <td className="cell-nowrap">{formatTime(item.timestamp)}</td>
                      {columns.app && <td>{item.data?.app_name ?? "—"}</td>}
                      {columns.title && <td className="cell-ellipsis">{item.data?.window_title ?? "—"}</td>}
                      {columns.process && <td>{item.data?.process_name ?? "—"}</td>}
                      {columns.duration && <td>{formatDuration(item.duration_s)}</td>}
                      {columns.status && <td>{item.data?.is_idle ? "空闲" : "活跃"}</td>}
                      {columns.raw && (
                        <td className="cell-raw">
                          {item.event_type} · {String(item.id).slice(0, 8)}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="changePage">
            <div className="page">{`共 ${total} 条，第 ${page} / ${totalPages} 页`}</div>
            <div className="button-box">
              <button
                type="button"
                className="btn btn-secondary btn-sm last-page"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1 || loading}
              >
                上一页
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm next-page"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages || loading}
              >
                下一页
              </button>
            </div>
          </div>
        </Surface>
      </div>
    </>
  );
}
