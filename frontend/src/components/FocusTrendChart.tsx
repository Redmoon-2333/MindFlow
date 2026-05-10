import { Card } from 'antd'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area } from 'recharts'
import { TrendDataPoint } from '../types'

interface FocusTrendChartProps {
  data: TrendDataPoint[]
}

export default function FocusTrendChart({ data }: FocusTrendChartProps) {
  return (
    <Card
      title="专注趋势 (近7天)"
      style={{ flex: 1, minWidth: 300 }}
    >
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}`}
            />
            <Tooltip
              formatter={(value: number, name: string) => {
                if (name === 'focus_score') return [value, '专注得分']
                return [value, '专注时长']
              }}
            />
            <Area
              type="monotone"
              dataKey="focus_score"
              stroke="#1677ff"
              strokeWidth={2}
              fill="url(#scoreGradient)"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="focus_score"
              stroke="#1677ff"
              strokeWidth={2}
              dot={{ r: 4, fill: '#1677ff', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 6, fill: '#1677ff' }}
            />
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1677ff" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#1677ff" stopOpacity={0} />
              </linearGradient>
            </defs>
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#8c8c8c' }}>
          暂无趋势数据
        </div>
      )}
    </Card>
  )
}
