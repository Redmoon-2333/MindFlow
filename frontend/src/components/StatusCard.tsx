import { Card, Statistic } from 'antd'
import { ActivityInfo } from '../types'
import { ClockCircleOutlined, SyncOutlined } from '@ant-design/icons'

interface StatusCardProps {
  activity: ActivityInfo | null
  running: boolean
}

export default function StatusCard({ activity, running }: StatusCardProps) {
  return (
    <Card
      title={
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: running ? '#52c41a' : '#ff4d4f',
              display: 'inline-block',
              flexShrink: 0,
            }}
          />
          实时状态
        </span>
      }
      style={{ marginBottom: 24 }}
    >
      {activity ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <div style={{ color: '#8c8c8c', fontSize: 13, marginBottom: 4 }}>
              当前窗口
            </div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 500,
                color: '#141414',
              }}
            >
              {activity.window_title || '(无标题)'}
            </div>
          </div>
          <div>
            <div style={{ color: '#8c8c8c', fontSize: 13, marginBottom: 4 }}>
              进程
            </div>
            <div style={{ fontFamily: 'monospace', fontSize: 14, color: '#595959' }}>
              {activity.process_name}
            </div>
          </div>
          <div>
            <div style={{ color: '#8c8c8c', fontSize: 13, marginBottom: 4 }}>
              更新时间
            </div>
            <div style={{ fontSize: 14, color: '#595959' }}>
              {new Date(activity.timestamp * 1000).toLocaleTimeString('zh-CN')}
            </div>
          </div>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '24px 0', color: '#8c8c8c' }}>
          <SyncOutlined style={{ fontSize: 24, marginBottom: 8 }} spin={running} />
          <div>{running ? '等待活动数据...' : '收集器未运行'}</div>
        </div>
      )}
    </Card>
  )
}
