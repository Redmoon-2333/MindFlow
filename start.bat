@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 > nul
set "ROOT=%~dp0"
set "FRONTEND_PORT=4173"

echo.
echo ╔═══════════════════════════════════╗
echo ║     MindFlow - 智能专注助手      ║
echo ╚═══════════════════════════════════╝
echo.
echo [1] 开发模式
echo     启动 backend-next + Vite，并打开 http://127.0.0.1:4173
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
call :check_port %FRONTEND_PORT%
if errorlevel 1 goto end
echo 正在启动后端...
start "MindFlow Backend" /D "%ROOT%backend-next" uv run python -m mindflow.main
call :wait_url http://127.0.0.1:8765/api/v1/health/live
if errorlevel 1 (
  echo 后端未能在预期时间内就绪，请检查 backend-next 日志。
  goto end
)
echo 正在启动 Vite...
start "MindFlow Frontend" /D "%ROOT%frontend" npm run dev -- --host 127.0.0.1 --port %FRONTEND_PORT%
call :wait_url http://127.0.0.1:%FRONTEND_PORT%/
if errorlevel 1 (
  echo Vite 未能在预期时间内就绪，请检查前端窗口。
  goto end
)
set "UI_URL="
for /f "usebackq delims=" %%U in (`cd /d "%ROOT%backend-next" ^& uv run python -m mindflow.bootstrap 2^>nul`) do set "UI_URL=%%U"
if defined UI_URL goto open_dev
echo 无法获取开发界面启动 URL，请检查后端日志。
goto end

:open_dev
set "UI_URL=!UI_URL:8765=%FRONTEND_PORT%!"
start "" "!UI_URL!"
set "UI_URL="
goto end

:production
echo.
call :wait_url http://127.0.0.1:8765/api/v1/health/live
if not errorlevel 1 goto production_ready
echo 正在启动后端...
start "MindFlow Backend" /D "%ROOT%backend-next" uv run python -m mindflow.main
echo 等待后端就绪并获取一次性启动 URL...
call :wait_url http://127.0.0.1:8765/api/v1/health/live
if errorlevel 1 (
  echo 后端未能在预期时间内就绪，请检查 backend-next 日志。
  goto end
)
:production_ready
set "UI_URL="
for /f "usebackq delims=" %%U in (`cd /d "%ROOT%backend-next" ^& uv run python -m mindflow.bootstrap 2^>nul`) do set "UI_URL=%%U"
if defined UI_URL goto open_production
echo 无法获取生产界面启动 URL，请检查后端日志。
goto end

:open_production
echo 正在打开 MindFlow...
start "" "!UI_URL!"
set "UI_URL="
goto end

:backend
cd /d "%ROOT%backend-next"
uv run python -m mindflow.main
goto end

:end
echo.
echo MindFlow 启动流程已完成。
pause
endlocal
exit /b 0

:check_port
netstat -ano | findstr /R /C:":%~1 .*LISTENING" > nul
if not errorlevel 1 (
  echo 端口 %~1 已被占用，请关闭占用它的程序后重试。
  exit /b 1
)
exit /b 0

:wait_url
for /l %%N in (1,1,30) do (
  curl.exe -fsS --max-time 2 "%~1" > nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 1 /nobreak > nul
)
exit /b 1
