@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 > nul
set "ROOT=%~dp0"

echo.
echo ╔═══════════════════════════════════╗
echo ║     MindFlow - 智能专注助手      ║
echo ╚═══════════════════════════════════╝
echo.
echo [1] 开发模式
echo     启动 backend-next + Vite，并打开 http://127.0.0.1:5173
echo.
echo [2] 生产界面
echo     启动后端，获取一次性启动 URL，并打开 8765 界面
echo.
echo [3] 仅启动后端
echo.
set /p choice="请选择 (1/2/3): "
if "%choice%"=="1" goto dev
if "%choice%"=="2" goto production
if "%choice%"=="3" goto backend
echo 无效选择，退出
pause
exit /b 1

:dev
echo.
echo 正在启动后端...
start "MindFlow Backend" /D "%ROOT%backend-next" python -m mindflow.main
echo 正在启动 Vite...
start "MindFlow Frontend" /D "%ROOT%frontend" npm run dev
timeout /t 3 /nobreak > nul
set "UI_URL="
for /l %%N in (1,1,20) do (
  for /f "usebackq delims=" %%U in (`cd /d "%ROOT%backend-next" ^& python -m mindflow.bootstrap 2^>nul`) do set "UI_URL=%%U"
  if defined UI_URL goto open_dev
  timeout /t 1 /nobreak > nul
)
echo 无法获取开发界面启动 URL，请检查后端日志。
goto end

:open_dev
set "UI_URL=!UI_URL:8765=5173!"
start "" "!UI_URL!"
set "UI_URL="
goto end

:production
echo.
echo 正在启动后端...
start "MindFlow Backend" /D "%ROOT%backend-next" python -m mindflow.main
echo 等待后端就绪并获取一次性启动 URL...
set "UI_URL="
for /l %%N in (1,1,20) do (
  for /f "usebackq delims=" %%U in (`cd /d "%ROOT%backend-next" ^& python -m mindflow.bootstrap 2^>nul`) do set "UI_URL=%%U"
  if defined UI_URL goto open_production
  timeout /t 1 /nobreak > nul
)
echo 无法获取生产界面启动 URL，请检查后端日志。
goto end

:open_production
echo 正在打开 MindFlow...
start "" "!UI_URL!"
set "UI_URL="
goto end

:backend
cd /d "%ROOT%backend-next"
python -m mindflow.main
goto end

:end
echo.
echo MindFlow 启动流程已完成。
pause
endlocal
