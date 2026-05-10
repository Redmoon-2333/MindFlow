import { Card, Row, Col, Statistic, Spin, Alert, Empty } from 'antd'
import { TrophyOutlined, ClockCircleOutlined } from '@ant-design/icons'
import { useApi } from '../hooks/useApi'
import { useWebSocket } from '../hooks/useWebSocket'
import type { TodayReport, TrendDataPoint, ActivityInfo } from '../types'
import StatusCard from './StatusCard'
import AppUsageChart from './AppUsageChart'
import FocusTrendChart from './FocusTrendChart'

export default function Dashboard() {
  const { data: todayData, loading: todayLoading, error: todayError } = useApi<TodayReport>('/focus/today')
  const { data: trendData, loading: trendLoading, error: trendError } = useApi<TrendDataPoint[]>('/focus/trend?days=7')
  const { latestActivity } = useWebSocket()

  const activity: ActivityInfo | null = latestActivity
    ? (() => {
        try {
          const parsed = JSON.parse(latestActivity)
          return {
            process_name: parsed.process_name || '',
            window_title: parsed.window_title || '',
            timestamp: parsed.timestamp || Math.floor(Date.now() / 1000),
          }
        } catch {
          return null
        }
      })()
    : null

  if (todayLoading || trendLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}>
        <Spin size="large" tip="加载数据..." />
      </div>
    )
  }

  if (todayError) {
    return (
      <div style={{ padding: 24 }}>
        <Alert message="获取今日数据失败" description={todayError} type="error" showIcon />
      </div>
    )
  }

  const report = todayData?.focus_report
  const status = todayData?.collector_status

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12}>
          <Card>
            <Statistic
              title="今日专注得分"
              value={report?.focus_score ?? 0}
              prefix={<TrophyOutlined />}
              suffix="/ 100"
              valueStyle={{ color: '#1677ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12}>
          <Card>
            <Statistic
              title="专注时长"
              value={report?.total_focus_minutes ?? 0}
              prefix={<ClockCircleOutlined />}
              suffix="分钟"
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {status && (
        <StatusCard
          activity={activity ?? status.current_activity}
          running={status.running}
        />
      )}

      <Row gutter={[16, 0]}>
        <Col xs={24} xl={12}>
          <AppUsageChart apps={report?.top_apps ?? []} />
        </Col>
        <Col xs={24} xl={12}>
          <FocusTrendChart data={trendData ?? []} />
        </Col>
      </Row>
    </div>
  )
}
