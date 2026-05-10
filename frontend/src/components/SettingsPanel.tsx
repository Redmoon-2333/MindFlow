import { useState, useEffect } from 'react'
import { Card, Form, InputNumber, Switch, Space, Alert, Button, Tag, Input } from 'antd'
import { useApi } from '../hooks/useApi'
import { Preferences } from '../types'
import { SaveOutlined } from '@ant-design/icons'

export default function SettingsPanel() {
  const [form] = Form.useForm()
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const { data: prefs, loading, error: fetchError } = useApi<Preferences>('/preferences')

  useEffect(() => {
    if (prefs) {
      form.setFieldsValue(prefs)
    }
  }, [prefs, form])

  const handleSave = async () => {
    const values = form.getFieldsValue()
    setSaving(true)
    setMessage(null)
    try {
      const res = await fetch('/api/v1/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setMessage({ type: 'success', text: '设置已保存' })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '保存失败' })
    } finally {
      setSaving(false)
    }
  }

  const handleToggleCollector = async (checked: boolean) => {
    setMessage(null)
    try {
      const res = await fetch('/api/v1/collector', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: checked ? 'start' : 'stop' }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setMessage({ type: 'success', text: checked ? '收集器已启动' : '收集器已停止' })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '操作失败' })
      form.setFieldsValue({ collector_enabled: !checked })
    }
  }

  if (fetchError) {
    return (
      <div style={{ padding: 24 }}>
        <Alert message="加载设置失败" description={fetchError} type="error" showIcon />
      </div>
    )
  }

  return (
    <div style={{ maxWidth: 640 }}>
      {message && (
        <Alert
          message={message.text}
          type={message.type}
          showIcon
          closable
          onClose={() => setMessage(null)}
          style={{ marginBottom: 24 }}
        />
      )}

      <Card title="收集器控制" style={{ marginBottom: 24 }}>
        <Space direction="vertical" size="middle">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontWeight: 500 }}>运行状态</div>
              <div style={{ color: '#8c8c8c', fontSize: 13 }}>控制活动收集器的启停</div>
            </div>
            <Switch
              checked={form.getFieldValue('collector_enabled')}
              onChange={handleToggleCollector}
              loading={loading}
            />
          </div>
        </Space>
      </Card>

      <Card title="专注设置" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            session_duration_minutes: 25,
            break_duration_minutes: 5,
            focus_apps: [],
            distraction_apps: [],
          }}
        >
          <Form.Item
            label="专注时长 (分钟)"
            name="session_duration_minutes"
          >
            <InputNumber min={1} max={120} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            label="休息时长 (分钟)"
            name="break_duration_minutes"
          >
            <InputNumber min={1} max={60} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item label="专注应用列表">
            <Input
              placeholder="多个应用用逗号分隔"
              value={form.getFieldValue('focus_apps')?.join(', ') || ''}
              onChange={(e) =>
                form.setFieldsValue({
                  focus_apps: e.target.value
                    .split(',')
                    .map((s) => s.trim())
                    .filter(Boolean),
                })
              }
            />
          </Form.Item>
          <Form.Item label="干扰应用列表">
            <Input
              placeholder="多个应用用逗号分隔"
              value={form.getFieldValue('distraction_apps')?.join(', ') || ''}
              onChange={(e) =>
                form.setFieldsValue({
                  distraction_apps: e.target.value
                    .split(',')
                    .map((s) => s.trim())
                    .filter(Boolean),
                })
              }
            />
          </Form.Item>
        </Form>
      </Card>

      <Button
        type="primary"
        icon={<SaveOutlined />}
        onClick={handleSave}
        loading={saving}
        size="large"
      >
        保存设置
      </Button>
    </div>
  )
}
