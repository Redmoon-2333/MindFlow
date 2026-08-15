import { useState, useEffect, useCallback } from "react";
import {
  getTasks,
  createTask,
  updateTask,
  deleteTask,
  getBlocklist,
  addBlockedSite,
  toggleBlockedSite,
  removeBlockedSite,
  getErrorMessage,
} from "../api";
import type { BlockedSiteItem, TaskItem, TaskListResponse } from "../api";

const STATUS_FILTERS = [
  { key: "all", label: "全部" },
  { key: "pending", label: "待办" },
  { key: "in_progress", label: "进行中" },
  { key: "done", label: "已完成" },
] as const;

type StatusFilter = (typeof STATUS_FILTERS)[number]["key"];

const STATUS_LABELS: Record<TaskItem["status"], string> = {
  pending: "待办",
  in_progress: "进行中",
  done: "已完成",
};

const STATUS_BADGES: Record<TaskItem["status"], string> = {
  pending: "badge-warning",
  in_progress: "badge-info",
  done: "badge-success",
};

function formatDeadline(deadlineUtc: string | null): string {
  if (!deadlineUtc) return "无截止";
  const date = new Date(deadlineUtc);
  if (Number.isNaN(date.getTime())) return deadlineUtc;
  return date.toLocaleString("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Convert a datetime-local input value (local time) to an ISO UTC string. */
function localInputToUtcIso(value: string): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

export default function Execution() {
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [tasksLoading, setTasksLoading] = useState(true);
  const [blocklist, setBlocklist] = useState<BlockedSiteItem[]>([]);
  const [blocklistLoading, setBlocklistLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Task form state
  const [taskTitle, setTaskTitle] = useState("");
  const [taskPriority, setTaskPriority] = useState(3);
  const [taskDeadline, setTaskDeadline] = useState("");
  const [taskMinutes, setTaskMinutes] = useState("");

  // Blocklist form state
  const [blockDomain, setBlockDomain] = useState("");
  const [blockReason, setBlockReason] = useState("");

  const loadTasks = useCallback(async () => {
    setTasksLoading(true);
    try {
      const data: TaskListResponse = await getTasks();
      setTasks(data.items);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "加载任务失败"));
    } finally {
      setTasksLoading(false);
    }
  }, []);

  const loadBlocklist = useCallback(async () => {
    setBlocklistLoading(true);
    try {
      const data = await getBlocklist();
      setBlocklist(data.items);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "加载拦截列表失败"));
    } finally {
      setBlocklistLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTasks();
    loadBlocklist();
  }, [loadTasks, loadBlocklist]);

  const handleAddTask = async () => {
    if (!taskTitle.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await createTask({
        title: taskTitle.trim(),
        priority: taskPriority,
        deadline_utc: localInputToUtcIso(taskDeadline),
        estimated_minutes: taskMinutes ? Number(taskMinutes) : undefined,
      });
      setTaskTitle("");
      setTaskDeadline("");
      setTaskMinutes("");
      setTaskPriority(3);
      await loadTasks();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "创建任务失败"));
    } finally {
      setBusy(false);
    }
  };

  const handleAdvanceTask = async (task: TaskItem) => {
    setBusy(true);
    setError(null);
    try {
      const next: TaskItem["status"] =
        task.status === "pending"
          ? "in_progress"
          : task.status === "in_progress"
            ? "done"
            : "pending";
      await updateTask(task.id, { status: next });
      await loadTasks();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "更新任务失败"));
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteTask = async (task: TaskItem) => {
    setBusy(true);
    setError(null);
    try {
      await deleteTask(task.id);
      await loadTasks();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "删除任务失败"));
    } finally {
      setBusy(false);
    }
  };

  const handleAddBlock = async () => {
    if (!blockDomain.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await addBlockedSite(blockDomain.trim(), blockReason.trim() || undefined);
      setBlockDomain("");
      setBlockReason("");
      await loadBlocklist();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "添加拦截域名失败"));
    } finally {
      setBusy(false);
    }
  };

  const handleToggleBlock = async (item: BlockedSiteItem) => {
    setBusy(true);
    setError(null);
    try {
      await toggleBlockedSite(item.domain, !item.enabled);
      await loadBlocklist();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "切换拦截状态失败"));
    } finally {
      setBusy(false);
    }
  };

  const handleRemoveBlock = async (item: BlockedSiteItem) => {
    setBusy(true);
    setError(null);
    try {
      await removeBlockedSite(item.domain);
      await loadBlocklist();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "移除拦截域名失败"));
    } finally {
      setBusy(false);
    }
  };

  const filteredTasks =
    statusFilter === "all"
      ? tasks
      : tasks.filter((task) => task.status === statusFilter);

  return (
    <div>
      <div className="header">
        <h1>干预执行</h1>
        <p>任务优先级数据源与网站拦截执行（浏览器扩展自动应用）</p>
      </div>

      {error && <div className="error-box mb16">{error}</div>}

      <div className="card mb24">
        <h3>任务管理</h3>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
          优先级建议（smart_prioritization）干预会读取这里的待办任务，按截止时间与优先级排序后写入提醒
        </p>

        <div className="flex gap8" style={{ marginBottom: 16, flexWrap: "wrap" }}>
          <input
            placeholder="任务标题（必填）"
            value={taskTitle}
            onChange={(e) => setTaskTitle(e.target.value)}
            style={{ flex: 2, minWidth: 200 }}
          />
          <select
            value={taskPriority}
            onChange={(e) => setTaskPriority(Number(e.target.value))}
            title="优先级"
          >
            {[1, 2, 3, 4, 5].map((p) => (
              <option key={p} value={p}>
                优先级 {p}
              </option>
            ))}
          </select>
          <input
            type="datetime-local"
            value={taskDeadline}
            onChange={(e) => setTaskDeadline(e.target.value)}
            title="截止时间（可选）"
          />
          <input
            placeholder="预计分钟（可选）"
            type="number"
            min={1}
            value={taskMinutes}
            onChange={(e) => setTaskMinutes(e.target.value)}
            style={{ width: 140 }}
          />
          <button
            type="button"
            className="btn"
            disabled={busy || !taskTitle.trim()}
            onClick={handleAddTask}
          >
            {busy && (
              <span className="spinner" style={{ width: 14, height: 14, margin: 0, borderWidth: 2 }} />
            )}
            添加任务
          </button>
        </div>

        <div className="tabs" style={{ marginBottom: 12 }}>
          {STATUS_FILTERS.map((filter) => (
            <button
              type="button"
              key={filter.key}
              className={`tab ${statusFilter === filter.key ? "active" : ""}`}
              onClick={() => setStatusFilter(filter.key)}
            >
              {filter.label}
            </button>
          ))}
        </div>

        {tasksLoading && <div className="spinner" />}
        {!tasksLoading && filteredTasks.length === 0 && (
          <p style={{ color: "var(--color-text-tertiary)", fontSize: 14 }}>
            暂无任务，添加一个开始使用优先级建议
          </p>
        )}
        {!tasksLoading &&
          filteredTasks.map((task) => (
            <div
              key={task.id}
              className="flex-between"
              style={{
                padding: "12px 0",
                borderBottom: "1px solid var(--color-border)",
                alignItems: "center",
                gap: 12,
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="flex gap8" style={{ alignItems: "center", flexWrap: "wrap" }}>
                  <span
                    style={{
                      textDecoration: task.status === "done" ? "line-through" : "none",
                      fontSize: 14,
                      fontWeight: 600,
                    }}
                  >
                    {task.title}
                  </span>
                  <span className={`badge badge-info`}>P{task.priority}</span>
                  <span className={`badge ${STATUS_BADGES[task.status]}`}>
                    {STATUS_LABELS[task.status]}
                  </span>
                  {task.estimated_minutes != null && (
                    <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                      约 {task.estimated_minutes} 分钟
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                  截止：{formatDeadline(task.deadline_utc)}
                </div>
              </div>
              <div className="flex gap8" style={{ flexShrink: 0 }}>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={busy}
                  onClick={() => handleAdvanceTask(task)}
                >
                  {task.status === "pending"
                    ? "开始"
                    : task.status === "in_progress"
                      ? "完成"
                      : "重开"}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-danger"
                  disabled={busy}
                  onClick={() => handleDeleteTask(task)}
                >
                  删除
                </button>
              </div>
            </div>
          ))}
      </div>

      <div className="card">
        <h3>网站拦截</h3>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
          环境优化（environment_optimization）干预触发时会自动把近期停留最久的干扰域名加入这里；
          浏览器扩展每分钟同步一次并通过 declarativeNetRequest 规则真实拦截
        </p>

        <div className="flex gap8" style={{ marginBottom: 16, flexWrap: "wrap" }}>
          <input
            placeholder="域名，如 bilibili.com（必填）"
            value={blockDomain}
            onChange={(e) => setBlockDomain(e.target.value)}
            style={{ flex: 2, minWidth: 200 }}
          />
          <input
            placeholder="拦截原因（可选）"
            value={blockReason}
            onChange={(e) => setBlockReason(e.target.value)}
            style={{ flex: 1, minWidth: 160 }}
          />
          <button
            type="button"
            className="btn btn-danger"
            disabled={busy || !blockDomain.trim()}
            onClick={handleAddBlock}
          >
            添加拦截
          </button>
        </div>

        {blocklistLoading && <div className="spinner" />}
        {!blocklistLoading && blocklist.length === 0 && (
          <p style={{ color: "var(--color-text-tertiary)", fontSize: 14 }}>
            暂无拦截域名
          </p>
        )}
        {!blocklistLoading &&
          blocklist.map((item) => (
            <div
              key={item.domain}
              className="flex-between"
              style={{
                padding: "12px 0",
                borderBottom: "1px solid var(--color-border)",
                alignItems: "center",
                gap: 12,
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="flex gap8" style={{ alignItems: "center", flexWrap: "wrap" }}>
                  <span style={{ fontSize: 14, fontWeight: 600 }}>{item.domain}</span>
                  <span className={`badge ${item.enabled ? "badge-danger" : "badge-info"}`}>
                    {item.enabled ? "拦截中" : "已暂停"}
                  </span>
                </div>
                {item.reason && (
                  <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                    {item.reason}
                  </div>
                )}
              </div>
              <div className="flex gap8" style={{ flexShrink: 0 }}>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={busy}
                  onClick={() => handleToggleBlock(item)}
                >
                  {item.enabled ? "暂停" : "恢复"}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-danger"
                  disabled={busy}
                  onClick={() => handleRemoveBlock(item)}
                >
                  移除
                </button>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
