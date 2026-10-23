import { useEffect, useState } from "react";
import { getAISettings, saveAISettings, deleteAISettings, getErrorMessage, ApiError, type AICredentialStatus } from "../api";

function configError(error: unknown, fallback: string) {
  return error instanceof ApiError && error.status === 404 ? "当前后端没有 AI 配置接口（404 Not Found）。请重启仓库中的 backend-next 后端，并确认前端连接的是该实例；此错误不代表 Key 无效。" : getErrorMessage(error, fallback);
}

export default function AISettings() {
  const [status, setStatus] = useState<AICredentialStatus | null>(null);
  const [service, setService] = useState<"deepseek" | "ecnu">("deepseek");
  const [model, setModel] = useState<"ecnu-max" | "ecnu-plus">("ecnu-max");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  function accept(data: AICredentialStatus) {
    setStatus(data); setService(data.service ?? "deepseek");
    setModel(data.model === "ecnu-plus" ? "ecnu-plus" : "ecnu-max");
  }
  useEffect(() => {
    let active = true;
    getAISettings().then(data => { if (active) accept(data); }).catch(e => { if (active) setError(configError(e, "AI 配置加载失败")); });
    return () => { active = false; };
  }, []);
  async function reload() {
    setBusy(true); setError("");
    try { accept(await getAISettings()); } catch (e) { setError(configError(e, "AI 配置加载失败")); } finally { setBusy(false); }
  }
  async function update(remove = false) {
    setBusy(true); setError(""); setNotice("");
    try {
      accept(remove ? await deleteAISettings() : await saveAISettings(key.trim(), service, model));
      setKey("");
      setNotice(remove ? "已移除当前服务在本机保存的 Key。请重启后端以重新加载配置。" : "服务与 Key 已保存到本机用户配置目录。请重启后端后使用 AI 功能；保存不验证密钥是否可调用。");
    } catch (e) { setError(configError(e, "AI 配置保存失败")); } finally { setBusy(false); }
  }
  const selectedMatchesSaved = service === (status?.service ?? "deepseek");
  return <div className="card">
    <h3>AI 服务配置</h3>
    <p className="workspace-note">选择 DeepSeek 或 ChatECNU（华东师范大学）。两家服务的 Key 分别保存到本机用户配置目录，不写入仓库或浏览器存储；切换服务或修改 Key 后需要重启后端。未配置时仍可使用已有本地模型与规则分析。</p>
    {error && <div className="error-box mb16" role="alert">{error}<button className="btn btn-ghost btn-sm" type="button" disabled={busy} onClick={reload}>重新加载配置</button></div>}
    {notice && <div className="info-box mb16" role="status">{notice}</div>}
    {status && <div className="workspace-actions mb24"><span className={`badge ${status.active_configured ? "badge-success" : "badge-info"}`}>{status.active_configured ? "当前进程已配置" : "当前进程未配置"}</span><span>当前进程服务：{status.active_provider ?? status.provider}</span><span>保存的服务：{status.provider} · {status.model}</span><span>{status.saved ? "本机已保存 Key" : "本机未保存 Key"}</span>{status.restart_required && <span className="badge badge-warning">等待重启生效</span>}</div>}
    {status?.environment_override && <div className="info-box mb16">启动环境变量优先于网页保存的配置。若重启后仍使用旧服务或 Key，请检查 MINDFLOW_LLM__SERVICE、对应服务的 Key 及模型环境变量。</div>}
    <div className="workspace-form">
      <label className="workspace-field">AI 服务<select aria-label="AI 服务" value={service} disabled={busy} onChange={e => { setService(e.target.value as "deepseek" | "ecnu"); setKey(""); setNotice(""); }}><option value="deepseek">DeepSeek</option><option value="ecnu">ChatECNU（华东师范大学）</option></select></label>
      {service === "ecnu" && <label className="workspace-field">ChatECNU 模型<select aria-label="ChatECNU 模型" value={model} disabled={busy} onChange={e => setModel(e.target.value as "ecnu-max" | "ecnu-plus")}><option value="ecnu-max">ecnu-max</option><option value="ecnu-plus">ecnu-plus</option></select></label>}
    </div>
    <p className="workspace-note">{service === "ecnu" ? "请填写 ChatECNU 的“我的令牌”中申请的 API Key。请求连接学校服务器 https://chat.ecnu.edu.cn/open/api/v1。" : "请填写 DeepSeek 平台签发的 API Key，请勿填入 ChatECNU Key。"}</p>
    <label className="workspace-field">{service === "ecnu" ? "ChatECNU" : "DeepSeek"} API Key<input type="password" aria-label={`${service === "ecnu" ? "ChatECNU" : "DeepSeek"} API Key`} autoComplete="new-password" spellCheck={false} value={key} disabled={busy} onChange={e => setKey(e.target.value)} placeholder="输入新的 Key，已有密钥不会回显" maxLength={512} /></label>
    <div className="workspace-actions mt16"><button type="button" className="btn" disabled={busy || !status || !key.trim()} onClick={() => update()}>保存 AI Key</button><button type="button" className="btn btn-danger" disabled={busy || !selectedMatchesSaved || !status?.saved} onClick={() => update(true)}>移除保存的 Key</button></div>
  </div>;
}
