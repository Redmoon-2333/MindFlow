import WorkspacePage from "../components/WorkspacePage";
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
  const [tab, setTab] = useState("任务管理");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [taskDescription, setTaskDescription] = useState("");
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
      const save = editingId ? (data: Parameters<typeof createTask>[0]) => updateTask(editingId, data) : createTask;
      if (taskMinutes && (!Number.isInteger(Number(taskMinutes)) || Number(taskMinutes) < 1 || Number(taskMinutes) > 100000)) throw new Error("预计分钟必须是 1–100000 的整数");
      await save({
        title: taskTitle.trim(),
        description: taskDescription.trim(),
        priority: taskPriority,
        deadline_utc: localInputToUtcIso(taskDeadline),
        estimated_minutes: taskMinutes ? Number(taskMinutes) : null,
      });
      setEditingId(null);
      setTaskDescription("");
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
      if (editingId === task.id) { setEditingId(null); setTaskTitle(""); setTaskDescription(""); setTaskDeadline(""); setTaskMinutes(""); setTaskPriority(3); }
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

  const handleEditTask = (task: TaskItem) => {
    setEditingId(task.id); setTaskTitle(task.title); setTaskDescription(task.description); setTaskPriority(task.priority);
    const date = task.deadline_utc ? new Date(task.deadline_utc) : null;
    setTaskDeadline(date ? new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0,16) : "");
    setTaskMinutes(task.estimated_minutes == null ? "" : String(task.estimated_minutes));
    document.getElementById("task-editor")?.scrollIntoView({ behavior: "smooth", block: "center" });
    document.getElementById("task-title")?.focus({preventScroll:true});
  };
  const filteredTasks =
    statusFilter === "all"
      ? tasks
      : tasks.filter((task) => task.status === statusFilter);

  return (
    <WorkspacePage title="干预执行" description="把建议落到行动：整理待办任务，或暂时屏蔽容易让你分心的网站。" className="execution-page">
      <div className="tabs" aria-label="执行功能">{["任务管理", "网站拦截"].map(name => <button type="button" key={name} className={`tab ${tab === name ? "active" : ""}`} aria-pressed={tab === name} onClick={() => setTab(name)}>{name}</button>)}</div>

      {error && <div className="error-box mb16">{error}</div>}

      {tab === "任务管理" && <div className="card mb24" id="task-editor">
        <h3>{editingId ? "编辑任务" : "任务管理"}</h3>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
          写下你准备完成的事情。专注提醒会参考截止时间与优先级，帮你选择下一步。优先级 5 为最高。
        </p>

        <div className="workspace-form">
          <label className="workspace-field workspace-full">任务标题<input
            id="task-title" maxLength={200}
            placeholder="任务标题（必填）"
            aria-label="任务标题"
            value={taskTitle}
            onChange={(e) => setTaskTitle(e.target.value)}
          /></label>
          <label className="workspace-field workspace-full">任务描述（可选）<textarea aria-label="任务描述" maxLength={2000} rows={2} value={taskDescription} onChange={e => setTaskDescription(e.target.value)} /></label>
          <label className="workspace-field">优先级<select
            value={taskPriority}
            onChange={(e) => setTaskPriority(Number(e.target.value))}
            title="优先级"
            aria-label="任务优先级"
          >
            {[1, 2, 3, 4, 5].map((p) => (
              <option key={p} value={p}>
                优先级 {p}
              </option>
            ))}
          </select></label>
          <label className="workspace-field">截止时间（可选）<input
            type="datetime-local"
            value={taskDeadline}
            onChange={(e) => setTaskDeadline(e.target.value)}
            title="截止时间（可选）"
            aria-label="任务截止时间"
          /></label>
          <label className="workspace-field">预计用时（分钟，可选）<input
            placeholder="预计分钟（可选）"
            aria-label="预计分钟"
            type="number"
            min={1}
            value={taskMinutes}
            onChange={(e) => setTaskMinutes(e.target.value)}
            max={100000} step={1}
          /></label>
          <div className="workspace-actions">
          <button
            type="button"
            className="btn"
            disabled={busy || !taskTitle.trim()}
            onClick={handleAddTask}
          >
            {busy && (
              <span className="spinner" style={{ width: 14, height: 14, margin: 0, borderWidth: 2 }} />
            )}
            {editingId ? "保存任务" : "添加任务"}
          </button>
          {editingId && <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => { setEditingId(null); setTaskTitle(""); setTaskDescription(""); setTaskDeadline(""); setTaskMinutes(""); setTaskPriority(3); }}>取消编辑</button>}
          </div>
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
              className="workspace-record"
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
                {task.description && <p className="workspace-note" style={{margin: "10px 0"}}>{task.description}</p>}
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                  截止：{formatDeadline(task.deadline_utc)}
                </div>
              </div>
              <div className="workspace-actions">
                <button type="button" className="btn btn-sm btn-ghost" disabled={busy} onClick={() => handleEditTask(task)}>编辑</button>
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
      </div>}

      {tab === "网站拦截" && <div className="card">
        <h3>网站拦截</h3>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
          先在系统设置中配对浏览器扩展，再添加想暂时屏蔽的域名。扩展每分钟同步名单；暂停拦截后可以恢复访问。
        </p>

        <div className="workspace-form">
          <input
            placeholder="域名，如 bilibili.com（必填）"
            aria-label="要拦截的域名"
            value={blockDomain}
            onChange={(e) => setBlockDomain(e.target.value)}
            style={{ minWidth: 0 }}
          />
          <input
            placeholder="拦截原因（可选）"
            aria-label="拦截原因"
            value={blockReason}
            onChange={(e) => setBlockReason(e.target.value)}
            style={{ minWidth: 0 }}
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
              className="workspace-record"
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
      </div>}
    </WorkspacePage>
  );
}
