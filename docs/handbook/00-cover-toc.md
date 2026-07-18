---
title: "MindFlow 智能专注助手"
subtitle: "从零构建全栈入门手册"
author: "胡淙煜 · 软件工程学院"
date: "2026-07-18"
lang: "zh-CN"
toc: true
toc-depth: 2
numbersections: true
mainfont: "SimSun"
monofont: "Consolas"
fontsize: 11pt
geometry: "margin=2.5cm"
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{MindFlow 全栈入门手册}
  - \fancyhead[R]{\thepage}
---

\thispagestyle{empty}

\begin{center}
\vspace*{4cm}

{\Huge \textbf{MindFlow}}\\[0.5cm]
{\LARGE 基于行为建模与 LLM 的智能专注助手}\\[1.5cm]

{\large \textbf{从零构建全栈入门手册}}\\[2cm]

{\large 胡淙煜}\\[0.3cm]
{\large 华东师范大学 软件工程学院}\\[2cm]

{\large 2026 年 7 月 18 日}\\[1cm]

{\small 版本: v3.0-langchain | 测试: 1000+ | 代码: MIT 开源}

\vfill
\end{center}

\newpage

# 序言

这本手册记录 MindFlow 后端从零到完整交付的全过程——从架构设计上的第一笔，到最后一行的部署命令。它不是一份API参考文档，而是一本"如果你要做一个类似的东西，可以从这里开始"的全栈指南。

**本书适合谁**: 前端队友（张皓、杨智杰）了解后端能力边界；未来的自己复盘设计决策；任何想理解"本地ML + 云端LLM 智能体"混合架构的人。

**如何阅读**: 按章节顺序最佳——每章依赖前一章建立的上下文。如果只想看某个模块：
- 想对前端 → 第6章
- 想理解 ML 算法 → 第3章
- 想理解为什么选 LangChain → 第5章
- 想跑起来 → 第1章 §1.6

**代码准确性承诺**: 本书中每段代码均来自 `backend-next/src/` 真实源文件，标注了精确的文件路径和行号。不包含伪代码或"示意代码"——所有代码片段都可以在仓库中找到对应的位置。

\newpage

# 目录
