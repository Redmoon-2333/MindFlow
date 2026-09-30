/**
 * Interface coverage matrix — page action → method/path → request/response →
 * auth → error/timeout → persisted-state verification.
 *
 * Everything goes through the Vite dev proxy (4173) so it lands on the
 * reviewer-owned backend (isolated temp data dir + port, no collectors, no
 * scheduler, no LLM credential). Nothing here touches a production database,
 * starts training, or enables automatic collection.
 *
 * Run: node tests/api-matrix.mjs [--base http://127.0.0.1:4173] [--rate-base <isolated default-rate server>] [--out <dir>]
 */
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

const args = process.argv.slice(2)
const argOf = (name, fallback) => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const BASE = argOf('base', 'http://127.0.0.1:4173')
const RATE_BASE = argOf('rate-base', BASE)
const OUT_DIR = path.resolve(argOf('out', '../docs/audit/20260930-repair'))

/** Rows: { group, action, method, path, req, expect, auth, timeoutMs, write } */
const rows = []
const record = (row) => rows.push(row)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** Perform a request, capturing status / latency / a body summary. */
async function call(method, urlPath, { cookie, body, headers = {}, timeoutMs = 30_000, origin = BASE } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const started = Date.now()
  try {
    const res = await fetch(`${origin}${urlPath}`, {
      method,
      headers: {
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(cookie ? { Cookie: cookie } : {}),
        ...headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    const text = await res.text()
    let json
    try { json = JSON.parse(text) } catch { json = undefined }
    return { status: res.status, ok: res.ok, ms: Date.now() - started, json, text, setCookie: res.headers.get('set-cookie'), retryAfter: res.headers.get('retry-after') }
  } catch (err) {
    return { status: 0, ok: false, ms: Date.now() - started, error: err?.name === 'AbortError' ? 'TIMEOUT' : String(err?.message ?? err) }
  } finally {
    clearTimeout(timer)
  }
}

/** Short human summary of a response body (never dump secrets). */
function summarise(res) {
  if (res.error) return res.error
  if (res.json === undefined) return (res.text ?? '').slice(0, 120)
  const j = res.json
  if (j === null || typeof j !== 'object') return JSON.stringify(j).slice(0, 120)
  if (typeof j.detail === 'string') return `detail=${j.detail.slice(0, 60)}`
  const keys = Object.keys(j)
  if (Array.isArray(j.items)) return `items[${j.items.length}] keys=${keys.slice(0, 6).join('|')}`
  return `keys=${keys.slice(0, 8).join('|')}`
}

// ── 0. Authentication handshake ────────────────────────────────────────────
let cookie = ''
{
  let ticket = null
  for (let attempt = 0; attempt < 6 && !ticket; attempt += 1) {
    const t = await call('POST', '/api/v1/auth/bootstrap/ticket')
    if (t.ok) ticket = t.json.ticket
    else if (t.status === 429) await sleep(2500)
    else break
  }
  const ex = await call('POST', '/api/v1/auth/bootstrap', { body: { ticket } })
  record({
    group: '认证',
    action: '登录（开发票据交换）',
    method: 'POST',
    path: '/api/v1/auth/bootstrap/ticket → /api/v1/auth/bootstrap',
    request: '{ticket} + Cookie',
    expect: '204 + Set-Cookie mindflow_session(HttpOnly,SameSite=Strict,Path=/api)',
    auth: '票据本身（代理注入根令牌）',
    result: `ticket=${ticket ? 'ok' : 'FAIL'} exchange=${ex.status}`,
    persist: '会话令牌入内存 SessionTokenStore（不落库）',
    ok: ex.status === 204,
  })
  const raw = ex.setCookie ?? ''
  const m = raw.match(/mindflow_session=([^;]*)/)
  cookie = m ? `mindflow_session=${m[1]}` : ''
}

// ── 1. Anonymous access is rejected ────────────────────────────────────────
{
  const anon = await call('GET', '/api/v1/preferences')
  record({
    group: '鉴权', action: '未带会话访问受保护接口', method: 'GET', path: '/api/v1/preferences',
    request: '（无 Cookie）', expect: '401 + RFC9457 problem', auth: '无',
    result: `${anon.status} ${summarise(anon)}`, persist: '—',
    ok: anon.status === 401,
  })
  const exempt = await call('GET', '/api/v1/health/live')
  record({
    group: '鉴权', action: '健康探针免鉴权', method: 'GET', path: '/api/v1/health/live',
    request: '（无 Cookie）', expect: '200（豁免路径）', auth: '豁免',
    result: `${exempt.status} ${summarise(exempt)}`, persist: '—', ok: exempt.ok,
  })
}

// ── 2. Read surface (page load) ───────────────────────────────────────────
const reads = [
  ['仪表盘', '健康状态', 'GET', '/api/v1/health', '200 {status,version,collector,database}'],
  ['仪表盘', '就绪探测', 'GET', '/api/v1/health/ready', '200 或 503（探针失败时）'],
  ['仪表盘', '采集器状态', 'GET', '/api/v1/collector', '200 {status,running}'],
  ['仪表盘', '自主状态', 'GET', '/api/v1/autonomy', '200 {enabled,paused_until,paused}'],
  ['仪表盘', 'LLM 配置状态', 'GET', '/api/v1/ai/provider-status', '200 {provider,model,configured,ollama_enabled}'],
  ['仪表盘', '模型状态', 'GET', '/api/v1/analytics/model-status', '200 {loaded,ready,mode}'],
  ['仪表盘', '专注预测', 'GET', '/api/v1/telemetry/focus-prediction', '200 六态之一 + focus_probability'],
  ['仪表盘', '当前活动', 'GET', '/api/v1/activities/current', '200 或 404（无记录）'],
  ['仪表盘', '近7日趋势', 'GET', '/api/v1/focus/trend?days=7', '200 {daily[]}'],
  ['仪表盘', '干预历史', 'GET', '/api/v1/intervention/history?days=7', '200 {items[],count}'],
  ['专注分析', '指定日期会话', 'GET', '/api/v1/focus?date=2026-09-30', '200 {date,sessions[],session_count}'],
  ['活动日志', '分页列表', 'GET', '/api/v1/activities?page=1&page_size=10', '200 {items,page,total,has_more}'],
  ['活动日志', '服务端搜索 q', 'GET', '/api/v1/activities?page=1&page_size=10&q=Code', '200 命中子集，total 与 items 同条件'],
  ['行为洞察', '模式分析', 'GET', '/api/v1/analytics/patterns?days=14', '200 {high_switch_periods,trigger_apps}'],
  ['行为洞察', '个人画像', 'GET', '/api/v1/analytics/profile?days=14', '200 {peak_focus_hours,top_apps,details}'],
  ['行为洞察', '基线', 'GET', '/api/v1/analytics/baseline', '200 或 404（尚无基线=业务常态）'],
  ['行为洞察', 'LLM 用量', 'GET', '/api/v1/analytics/usage', '200 {mode,llm_calls_30d}'],
  ['报告中心', '日报', 'GET', '/api/v1/reports/daily?date=2026-09-30', '200 + data_state'],
  ['报告中心', '周报', 'GET', '/api/v1/reports/weekly?week_start=2026-09-28', '200 + data_state'],
  ['干预中心', '历史列表', 'GET', '/api/v1/intervention/history?days=7', '200 {items[],count}'],
  ['专家面板', '上次结果', 'GET', '/api/v1/panel', '200 {types,confidence} 或 空态'],
  ['AI 对话', '会话列表', 'GET', '/api/v1/chat/sessions', '200 [ ]'],
  ['AI 对话', '历史消息', 'GET', '/api/v1/chat/{session_id}/messages', '200/404'],
  ['系统设置', '偏好', 'GET', '/api/v1/preferences', '200 {…}'],
  ['系统设置', '分类规则', 'GET', '/api/v1/app-classifications', '200 [ ]'],
  ['系统设置', '未知应用', 'GET', '/api/v1/app-classifications/unknown-apps', '200 [ ]'],
  ['系统设置', '遥测状态', 'GET', '/api/v1/telemetry/status', '200 {preferences,database_size_bytes}'],
  ['干预执行', '任务列表', 'GET', '/api/v1/tasks', '200 {items,total}'],
  ['干预执行', '屏蔽列表', 'GET', '/api/v1/interventions/blocklist', '200 {items}'],
  ['模型中心', '训练就绪度', 'GET', '/api/v1/analytics/training-readiness', '200 + quality_gates'],
  ['AI 诊断', '工作流运行', 'GET', '/api/v1/ai/runs?limit=10&offset=0', '200 {items,count,has_more}'],
  ['AI 诊断', '运行详情', 'GET', '/api/v1/ai/runs/__missing__', '404 RFC9457'],
  ['AI 诊断', '分析图拓扑', 'GET', '/api/v1/ai/graph', '200 {nodes,edges,available}'],
  ['全局', '导出 JSON', 'GET', '/api/v1/export?fmt=json', '200 流式导出'],
]
for (const [group, action, method, path, expect] of reads) {
  const res = await call(method, path, { cookie })
  const permitsNotFound = [
    '/api/v1/activities/current', '/api/v1/analytics/baseline', '/api/v1/panel',
    '/api/v1/chat/{session_id}/messages', '/api/v1/ai/runs/__missing__',
  ].includes(path)
  const validPanelEmpty = path !== '/api/v1/panel' || res.status !== 404
    || (res.json?.status === 404 && res.json?.detail?.includes('今日尚无面板分析结果'))
  const expectedStatus = res.status === 200
    || (permitsNotFound && res.status === 404 && res.json?.status === 404)
    || (path === '/api/v1/health/ready' && res.status === 503)
  record({
    group, action, method, path, request: 'Cookie 会话', expect,
    auth: '会话 Cookie',
    result: `${res.status} ${summarise(res)}`,
    persist: '只读', ok: expectedStatus && validPanelEmpty,
  })
}

// ── 3. Write surface + persisted-state verification ───────────────────────
async function writeThenRead({ group, action, method, path, body, expect, readMethod, readPath, verify, note }) {
  const created = await call(method, path, { cookie, body })
  let readBack = null
  let found
  if (created.ok && readPath) {
    readBack = await call(readMethod ?? 'GET', readPath, { cookie })
    found = verify ? verify(readBack) : undefined
  } else if (created.ok && verify) {
    found = verify(created)
  }
  record({
    group, action, method, path,
    request: JSON.stringify(body ?? {}).slice(0, 80),
    expect, auth: '会话 Cookie',
    result: `${created.status} ${summarise(created)}${readBack ? ` | read-back ${readBack.status} ${summarise(readBack)}` : ''}`,
    persist: !readBack ? '未做读回，仅校验响应' : (found ? `${note ?? '落库已确认'}：读回命中` : '读回未命中（失败）'),
    ok: created.ok && (!readBack || readBack.ok) && (found === undefined || found),
  })
  return created
}

const newRule = { process_name: `matrix_${Date.now()}.exe`, window_title_pattern: null, category: 'other', priority: 10 }
const createdRule = await writeThenRead({
  group: '系统设置', action: '新增分类规则', method: 'POST', path: '/api/v1/app-classifications',
  body: newRule, expect: '201 + 新建规则',
  readPath: '/api/v1/app-classifications',
  verify: (r) => Array.isArray(r.json) && r.json.some((x) => x.process_name === newRule.process_name),
  note: 'SQLite app_classification_rules',
})
const ruleId = createdRule.json?.id

if (ruleId) {
  const del = await call('DELETE', `/api/v1/app-classifications/${ruleId}`, { cookie })
  const after = await call('GET', '/api/v1/app-classifications', { cookie })
  const gone = after.ok && Array.isArray(after.json) && !after.json.some((x) => x.id === ruleId)
  record({
    group: '系统设置', action: '删除分类规则', method: 'DELETE', path: `/api/v1/app-classifications/{id}`,
    request: 'path id', expect: '200/204 且读回不再出现', auth: '会话 Cookie',
    result: `${del.status} | read-back ${after.status}`,
    persist: gone ? '落库删除已确认：读回不含该规则' : '读回仍含该规则（失败）', ok: del.ok && gone,
  })
}

await writeThenRead({
  group: '系统设置', action: '全量更新偏好', method: 'PUT', path: '/api/v1/preferences',
  body: { theme: 'dark' }, expect: '200 更新后对象',
  readPath: '/api/v1/preferences',
  verify: (r) => r.json?.theme === 'dark',
  note: 'SQLite user_preferences',
})
await writeThenRead({
  group: '系统设置', action: '局部更新偏好', method: 'PATCH', path: '/api/v1/preferences',
  body: { theme: 'light' }, expect: '200 合并结果',
  readPath: '/api/v1/preferences',
  verify: (r) => r.json?.theme === 'light',
  note: 'SQLite user_preferences',
})
await writeThenRead({
  group: '系统设置', action: '修改遥测偏好', method: 'PATCH', path: '/api/v1/telemetry/preferences',
  body: { input_telemetry_enabled: false }, expect: '200 更新后偏好',
  readPath: '/api/v1/telemetry/status',
  verify: (r) => r.json?.preferences?.input_telemetry_enabled === false,
  note: 'SQLite user_preferences',
})
await writeThenRead({
  group: '系统设置', action: '生成浏览器配对码', method: 'POST', path: '/api/v1/telemetry/browser/pairing-code',
  expect: '200 {code,expires_at}',
  verify: (r) => typeof r.json?.code === 'string',
  note: '配对码入内存/表，含过期时间',
})

const task = { title: `matrix task ${Date.now()}`, description: 'interface matrix', priority: 3, status: 'pending' }
const createdTask = await writeThenRead({
  group: '干预执行', action: '新建任务', method: 'POST', path: '/api/v1/tasks',
  body: task, expect: '201 + 任务对象',
  readPath: '/api/v1/tasks',
  verify: (r) => Array.isArray(r.json?.items) && r.json.items.some((item) => item.title === task.title),
  note: 'SQLite tasks',
})
const taskId = createdTask.json?.id
if (taskId) {
  const patched = await call('PATCH', `/api/v1/tasks/${taskId}`, { cookie, body: { status: 'done' } })
  const reread = await call('GET', '/api/v1/tasks', { cookie })
  const seenDone = Array.isArray(reread.json?.items) && reread.json.items.some((t) => t.id === taskId && t.status === 'done')
  record({
    group: '干预执行', action: '更新任务状态', method: 'PATCH', path: '/api/v1/tasks/{id}',
    request: '{"status":"done"}', expect: '200 且读回为 done', auth: '会话 Cookie',
    result: `${patched.status} ${summarise(patched)}`,
    persist: seenDone ? '落库已确认：读回 status=done' : '读回未更新（失败）',
    ok: patched.ok && seenDone,
  })
  const delTask = await call('DELETE', `/api/v1/tasks/${taskId}`, { cookie })
  const afterTasks = await call('GET', '/api/v1/tasks', { cookie })
  const taskGone = afterTasks.ok && Array.isArray(afterTasks.json?.items) && !afterTasks.json.items.some((t) => t.id === taskId)
  record({
    group: '干预执行', action: '删除任务', method: 'DELETE', path: '/api/v1/tasks/{id}',
    request: 'path id', expect: '200 且读回不再出现', auth: '会话 Cookie',
    result: `${delTask.status} | read-back ${afterTasks.status}`,
    persist: taskGone ? '落库删除已确认' : '读回仍存在（失败）', ok: delTask.ok && taskGone,
  })
}

const domain = `matrix-${Date.now()}.example`
await writeThenRead({
  group: '干预执行', action: '新增屏蔽域名', method: 'POST', path: '/api/v1/interventions/blocklist',
  body: { domain, reason: 'interface matrix' }, expect: '200 + 列表',
  readPath: '/api/v1/interventions/blocklist',
  verify: (r) => Array.isArray(r.json?.items) && r.json.items.some((b) => b.domain === domain),
  note: 'SQLite blocked_sites',
})
{
  const patch = await call('PATCH', `/api/v1/interventions/blocklist/${domain}`, { cookie, body: { enabled: false } })
  const reread = await call('GET', '/api/v1/interventions/blocklist', { cookie })
  const off = Array.isArray(reread.json?.items) && reread.json.items.some((b) => b.domain === domain && b.enabled === false)
  record({
    group: '干预执行', action: '停用屏蔽域名', method: 'PATCH', path: '/api/v1/interventions/blocklist/{domain}',
    request: '{"enabled":false}', expect: '200 且读回 enabled=false', auth: '会话 Cookie',
    result: `${patch.status} ${summarise(patch)}`,
    persist: off ? '落库已确认：读回 enabled=false' : '读回未更新（失败）', ok: patch.ok && off,
  })
  const del = await call('DELETE', `/api/v1/interventions/blocklist/${domain}`, { cookie })
  const reread2 = await call('GET', '/api/v1/interventions/blocklist', { cookie })
  const gone = reread2.ok && Array.isArray(reread2.json?.items) && !reread2.json.items.some((b) => b.domain === domain)
  record({
    group: '干预执行', action: '删除屏蔽域名', method: 'DELETE', path: '/api/v1/interventions/blocklist/{domain}',
    request: 'path domain', expect: '200 且读回不再出现', auth: '会话 Cookie',
    result: `${del.status} | read-back ${reread2.status}`,
    persist: gone ? '落库删除已确认' : '读回仍存在（失败）', ok: del.ok && gone,
  })
}

await writeThenRead({
  group: '仪表盘', action: '暂停自主干预', method: 'POST', path: '/api/v1/autonomy/pause',
  body: { hours: 1 }, expect: '200 paused_until 约 1 小时后',
  readPath: '/api/v1/autonomy',
  verify: (r) => r.json?.paused_until != null || r.json?.paused === true,
  note: 'user_preferences.autonomy',
})
{
  const res = await call('POST', '/api/v1/autonomy/resume', { cookie, body: {} })
  const reread = await call('GET', '/api/v1/autonomy', { cookie })
  const cleared = reread.json && (reread.json.paused_until === null || reread.json.paused === false)
  record({
    group: '仪表盘', action: '恢复自主干预', method: 'POST', path: '/api/v1/autonomy/resume',
    request: '{}', expect: '200 paused_until=null', auth: '会话 Cookie',
    result: `${res.status} ${summarise(res)} | read-back ${reread.status}`,
    persist: cleared ? '落库已确认：paused_until 已清除' : '读回仍暂停（失败）',
    ok: res.ok && cleared,
  })
}

await writeThenRead({
  group: '干预中心', action: '手动触发干预', method: 'POST', path: '/api/v1/intervention/trigger',
  body: { intensity: 'gentle' }, expect: '200 {intervention} 或 {skipped,skip_reason}',
  verify: (r) => r.json?.intervention !== undefined || r.json?.skipped !== undefined,
  note: '干预可能被节流跳过（业务态，非错误）',
})

{
  const date = '2026-09-30'
  const sessions = await call('GET', `/api/v1/focus?date=${date}`, { cookie })
  const session = sessions.json?.sessions?.[0]
  if (session?.id) {
    await writeThenRead({
      group: '专注分析', action: '保存专注反馈并读回', method: 'POST',
      path: `/api/v1/focus/${session.id}/feedback`,
      body: { label: 'focus', score: 5, task_type: 'coding' },
      expect: '200，读回同一会话的 label/score/task_type',
      readPath: `/api/v1/focus?date=${date}`,
      verify: (r) => r.json?.sessions?.some((item) => item.id === session.id
        && item.feedback_label === 'focus' && item.feedback_score === 5
        && item.feedback_task_type === 'coding'),
      note: '隔离数据库专注反馈',
    })
  } else {
    record({ group: '专注分析', action: '保存专注反馈', method: 'POST',
      path: '/api/v1/focus/{session_id}/feedback', result: '隔离库无会话，未执行', ok: null })
  }
}

// Duplicate / replay protection
{
  const t = await call('POST', '/api/v1/auth/bootstrap/ticket', { cookie })
  const first = await call('POST', '/api/v1/auth/bootstrap', { body: { ticket: t.json?.ticket } })
  const replay = await call('POST', '/api/v1/auth/bootstrap', { body: { ticket: t.json?.ticket } })
  record({
    group: '幂等/重复提交', action: '一次性票据重放', method: 'POST', path: '/api/v1/auth/bootstrap',
    request: '同一 ticket 提交两次', expect: '第一次 204，重放 401', auth: '票据',
    result: `first=${first.status} replay=${replay.status} ${summarise(replay)}`,
    persist: '票据不落库（consume 即删）', ok: first.status === 204 && replay.status === 401,
  })
}
{
  const payload = { process_name: `dup_${Date.now()}.exe`, window_title_pattern: null, category: 'other', priority: 1 }
  const a = await call('POST', '/api/v1/app-classifications', { cookie, body: payload })
  const b = await call('POST', '/api/v1/app-classifications', { cookie, body: payload })
  const list = await call('GET', '/api/v1/app-classifications', { cookie })
  const ids = Array.isArray(list.json) ? list.json.filter((x) => x.process_name === payload.process_name).map((x) => x.id) : []
  for (const id of ids) await call('DELETE', `/api/v1/app-classifications/${id}`, { cookie })
  record({
    group: '幂等/重复提交', action: '同一规则连续提交两次', method: 'POST', path: '/api/v1/app-classifications',
    request: '相同 body 两次', expect: '两次均 201（接口非幂等，客户端需自行去重）', auth: '会话 Cookie',
    result: `first=${a.status} second=${b.status} duplicates_in_db=${ids.length}`,
    persist: `清理前命中 ${ids.length} 条（已清理）`, ok: a.ok && b.ok,
    note: '记录为已知非幂等行为，不在本轮改动范围内',
  })
}

// ── 4. Error / validation surface ─────────────────────────────────────────
const errorProbes = [
  ['活动日志', '非法日期参数', 'GET', '/api/v1/activities?start_date=not-a-date', 422, '日期格式校验 RFC9457'],
  ['活动日志', '起止倒置', 'GET', '/api/v1/activities?start_date=2025-01-01&end_date=2024-01-01', 422, '业务校验'],
  ['活动日志', '非法游标', 'GET', '/api/v1/activities?cursor=not-a-cursor', 422, '游标校验'],
  ['活动日志', 'page=0', 'GET', '/api/v1/activities?page=0', 422, '分页参数校验'],
  ['干预执行', '空标题任务', 'POST', '/api/v1/tasks', 422, 'min_length 校验', { title: '' }],
  ['仪表盘', '低于下限暂停时长', 'POST', '/api/v1/autonomy/pause', 422, 'hours>=0.5', { hours: 0.1 }],
  ['AI 诊断', '不存在的运行', 'GET', '/api/v1/ai/runs/no-such-run', 404, 'RFC9457 not-found'],
  ['干预执行', '不存在的任务', 'DELETE', '/api/v1/tasks/no-such-id', 404, 'RFC9457 not-found'],
  ['报告中心', '未来日期', 'GET', '/api/v1/reports/daily?date=2099-01-01', 200, '返回 data_state=future（业务态，非 HTTP 错误）'],
  ['专家面板', '读取当前面板状态', 'GET', '/api/v1/panel', [200, 404], '已有结果=200，今日尚无结果=404 业务空态'],
]
for (const [group, action, method, path, expected, expect, body] of errorProbes) {
  const res = await call(method, path, { cookie, body })
  const expectedStatuses = Array.isArray(expected) ? expected : [expected]
  const validPanelEmpty = path !== '/api/v1/panel' || res.status !== 404
    || (res.json?.status === 404 && res.json?.detail?.includes('今日尚无面板分析结果'))
  record({
    group, action, method, path,
    request: body ? JSON.stringify(body) : '—',
    expect: `${expectedStatuses.join('/')} — ${expect}`, auth: '会话 Cookie',
    result: `${res.status} ${summarise(res)}`, persist: '—',
    ok: expectedStatuses.includes(res.status) && validPanelEmpty,
  })
}

// Rate limit (429) then retry
{
  const codes = []
  let retryAfter = null
  for (let i = 0; i < 130; i += 1) {
    const r = await call('GET', '/api/v1/health/live', { origin: RATE_BASE })
    codes.push(r.status)
    if (r.status === 429) {
      retryAfter = Number(r.retryAfter)
      break
    }
  }
  const hit429 = codes.includes(429)
  let retried = null
  if (hit429) {
    await sleep((retryAfter + 1) * 1000)
    const r = await call('GET', '/api/v1/health/live', { origin: RATE_BASE })
    retried = r.status
  }
  record({
    group: '限流', action: '突发请求触发全局令牌桶', method: 'GET', path: '/api/v1/health/live',
    request: '连续 130 次', expect: '429 + Retry-After，退避后恢复 200', auth: '健康探针免鉴权',
    result: `首个429=${hit429} Retry-After=${retryAfter} 退避后=${retried}`,
    persist: '—', ok: hit429 && retryAfter >= 1 && retried === 200,
    note: `默认限流隔离实例 ${RATE_BASE}；${hit429 ? '确认限流真实生效且可恢复' : '未触发429（桶未打空）'}`,
  })
}

// Client-side timeout (no backend fault injection available)
{
  const slow = await call('POST', '/api/v1/chat', { cookie, body: { message: 'ping' }, timeoutMs: 1 })
  record({
    group: '超时', action: '客户端极短超时中断', method: 'POST', path: '/api/v1/chat',
    request: '{"message":"ping"} + AbortSignal(1ms)', expect: '客户端 AbortError（不产生假成功）',
    auth: '会话 Cookie', result: `${slow.status === 0 ? 'aborted' : slow.status} ${slow.error ?? summarise(slow)}`,
    persist: '—', ok: slow.status === 0 && slow.error === 'TIMEOUT',
  })
}

// Real (degraded, offline) chat call — no LLM credential, so no external cost
{
  const res = await call('POST', '/api/v1/chat', { cookie, body: { message: '你好' }, timeoutMs: 60_000 })
  record({
    group: 'AI 对话', action: '发送消息（无凭证=降级）', method: 'POST', path: '/api/v1/chat',
    request: '{"message":"你好"}', expect: '200 {answer,degraded}；无凭证时走规则引擎，无外部调用',
    auth: '会话 Cookie', result: `${res.status} ${summarise(res)}`,
    persist: 'SQLite chat_messages（会话消息）', ok: res.ok,
    note: '真实 LLM 未配置 → 未验证在线推理',
  })
}

// Panel (rule-engine fallback, no external call)
{
  const res = await call('POST', '/api/v1/panel/today', { cookie, body: {}, timeoutMs: 90_000 })
  record({
    group: '专家面板', action: '运行面板（无凭证=降级）', method: 'POST', path: '/api/v1/panel/today',
    request: '{}', expect: '200 {types,source,degraded}；无凭证时降级到规则引擎',
    auth: '会话 Cookie', result: `${res.status} ${summarise(res)}`,
    persist: 'SQLite procrastination_analyses', ok: res.ok,
    note: '真实 LLM 未配置 → 未验证在线推理',
  })
}

// Expired session: log out, then reuse the old cookie
{
  const afterLogout = await call('POST', '/api/v1/auth/logout', { cookie })
  const reuse = await call('GET', '/api/v1/preferences', { cookie })
  record({
    group: '会话生命周期', action: '退出后复用旧会话', method: 'POST→GET', path: '/api/v1/auth/logout → /api/v1/preferences',
    request: '同一 Cookie', expect: '204 后复用返回 401', auth: '会话 Cookie（已撤销）',
    result: `logout=${afterLogout.status} reuse=${reuse.status} ${summarise(reuse)}`,
    persist: '服务端撤销该会话（其余会话不受影响）+ Cookie 过期',
    ok: afterLogout.status === 204 && reuse.status === 401,
  })
  // Re-establish a session for any later rows
  const t = await call('POST', '/api/v1/auth/bootstrap/ticket', {})
  if (t.ok) {
    const ex = await call('POST', '/api/v1/auth/bootstrap', { body: { ticket: t.json.ticket } })
    const c = ex.setCookie?.match(/mindflow_session=([^;]*)/)
    if (c) cookie = `mindflow_session=${c[1]}`
  }
}

// ── Deliberately NOT executed ─────────────────────────────────────────────
const skippedOps = [
  ['模型中心', '启动训练任务', 'POST', '/api/v1/analytics/training-jobs', '未执行 — 计划禁止在测试环境训练'],
  ['模型中心', '取消训练任务', 'POST', '/api/v1/analytics/training-jobs/{id}/cancel', '未执行 — 依赖训练任务'],
  ['仪表盘/设置', '启动采集器', 'POST', '/api/v1/collector', '未执行 — 计划禁止测试环境自动采集'],
  ['仪表盘/设置', '停止采集器', 'POST', '/api/v1/collector/stop', '未执行 — 对应启动未执行（保持 stopped）'],
  ['数据导出', '删除遥测数据', 'DELETE', '/api/v1/telemetry/data?scope=', '未执行 — 删除类操作留待用户确认'],
  ['全部', '注入 500/403', '—', '—', '未执行 — 本矩阵不注入服务故障；错误分支另由隔离回归覆盖'],
]
for (const [group, action, method, path, reason] of skippedOps) {
  record({ group, action, method, path, request: '—', expect: '—', auth: '—', result: reason, persist: '—', ok: null })
}

// ── Report ────────────────────────────────────────────────────────────────
await mkdir(OUT_DIR, { recursive: true })
const json = {
  generatedAt: new Date().toISOString(),
  base: BASE,
  executed: rows.filter((r) => r.ok !== null).length,
  passed: rows.filter((r) => r.ok === true).length,
  failed: rows.filter((r) => r.ok === false).length,
  skipped: rows.filter((r) => r.ok === null).length,
  rows,
}
await writeFile(path.join(OUT_DIR, 'api-matrix.json'), JSON.stringify(json, null, 2), 'utf8')

const groups = [...new Set(rows.map((r) => r.group))]
let md = '# 接口覆盖矩阵（页面操作 → 方法/路径 → 请求/响应 → 鉴权 → 错误/超时 → 落库）\n\n'
md += `- 生成时间：${json.generatedAt}\n`
md += `- 目标：\`${BASE}\`（Vite 代理 → reviewer-owned 后端，隔离数据目录/端口，无采集、无调度、无 LLM 凭证）\n`
md += `- 执行 ${json.executed} 项：**通过 ${json.passed} / 失败 ${json.failed}**，另有 ${json.skipped} 项按计划跳过\n\n`
md += '> 401/404/422/429/超时/重复提交/会话过期均按功能风险逐项覆盖；跳过项给出理由，不用 200 冒充覆盖。\n\n'
for (const g of groups) {
  md += `## ${g}\n\n| 操作 | 方法 | 路径 | 请求 | 期望 | 鉴权 | 实测 | 落库/副作用 | 判定 |\n|---|---|---|---|---|---|---|---|---|\n`
  for (const r of rows.filter((x) => x.group === g)) {
    const verdict = r.ok === null ? '跳过' : r.ok ? '✅' : '❌'
    const esc = (s) => String(s ?? '—').replace(/\|/g, '\\|').replace(/\n/g, ' ')
    md += `| ${esc(r.action)} | ${esc(r.method)} | \`${esc(r.path)}\` | ${esc(r.request)} | ${esc(r.expect)} | ${esc(r.auth)} | ${esc(r.result)} | ${esc(r.persist)} | ${verdict} |\n`
  }
  md += '\n'
}
await writeFile(path.join(OUT_DIR, 'api-matrix.md'), md, 'utf8')

console.log(`executed=${json.executed} passed=${json.passed} failed=${json.failed} skipped=${json.skipped}`)
for (const r of rows.filter((x) => x.ok === false)) {
  console.log(`  FAIL [${r.group}] ${r.method} ${r.path} → ${r.result}`)
}
console.log(`\nwrote ${path.join(OUT_DIR, 'api-matrix.md')}`)
process.exitCode = json.failed > 0 ? 1 : 0
