@echo off
chcp 65001 > nul
cd /d "%~dp0backend"

echo.
echo ╔═══════════════════════════════════╗
echo ║     MindFlow - 智能专注助手      ║
echo ╚═══════════════════════════════════╝
echo.
echo [1] 系统托盘模式（推荐）
echo     后台运行，右键托盘图标操作
echo.
echo [2] 浏览器模式
echo     命令行启动，浏览器打开 Dashboard
echo.
echo [3] 仅启动后端
echo     只启动 API 服务，不打开界面
echo.
set /p choice="请选择 (1/2/3): "

if "%choice%"=="1" goto tray
if "%choice%"=="2" goto browser
if "%choice%"=="3" goto backend
echo 无效选择，退出
pause
exit /b

:tray
echo.
echo 正在启动系统托盘...
python -m mindflow.tray
goto end

:browser
echo.
echo 正在启动后端...
start "" /B uvicorn mindflow.main:app --host 127.0.0.1 --port 8765
echo 等待后端就绪...
timeout /t 3 /nobreak > nul
echo 正在打开 Dashboard...
start http://127.0.0.1:8765/docs
echo.
echo 后端已启动，关闭此窗口不会停止后端
echo 需要停止时请关闭 uvicorn 窗口
goto end

:backend
echo.
echo 正在启动后端...
uvicorn mindflow.main:app --reload --host 127.0.0.1 --port 8765
goto end

:end
pause
