import { Card } from 'antd'
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import type { PieProps } from 'recharts'
import { AppUsageItem } from '../types'

const COLORS = [
  '#1677ff', '#4096ff', '#69b1ff', '#91caff',
  '#52c41a', '#73d13d', '#95de64', '#1677ff',
]

interface AppUsageChartProps {
  apps: AppUsageItem[]
}

const activeDot: PieProps['activeDot'] = (props: PieProps['activeDot'] extends infer U ? U : never) => {
  const { cx, cy, id } = props
  return <circle cx={cx} cy={cy} r={100} fill="transparent" key={id} />
}

export default function AppUsageChart({ apps }: AppUsageChartProps) {
  const data = apps
    .slice(0, 8)
    .map((item, index) => ({
      name: item.app,
      value: item.minutes,
      fill: COLORS[index % COLORS.length],
    }))

  return (
    <Card
      title="应用使用分布"
      style={{ flex: 1, minWidth: 300 }}
    >
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height={320}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={120}
              paddingAngle={3}
              dataKey="value"
              label={({ name, percent }: { name: string; percent: number }) =>
                `${name} ${(percent * 100).toFixed(0)}%`
              }
              activeDot={activeDot}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(1)} 分钟`, '时长']}
            />
            <Legend layout="vertical" verticalAlign="middle" align="right" />
          </PieChart>
        </ResponsiveContainer>
      ) : (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#8c8c8c' }}>
          暂无应用使用数据
        </div>
      )}
    </Card>
  )
}
