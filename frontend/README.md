# MindFlow Frontend

前端由张皓、杨智杰负责搭建。

## 技术栈

- React 19 + TypeScript
- Vite
- Ant Design
- recharts

## 启动

```bash
cd frontend
npm install
npm run dev
```

## 后端API

后端运行在 `http://localhost:8765`，API文档: `http://localhost:8765/docs`

Vite配置中已设置proxy，前端请求 `/api/*` 和 `/ws/*` 会自动转发到后端。

## 页面规划

参考 `../docs/design-spec.md` 第2节和第5节：
- Dashboard: 实时专注状态、应用使用饼图、专注趋势图
- Settings: 采集开关、干预强度配置
