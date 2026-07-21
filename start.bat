@echo off
chcp 65001 > nul
cd /d "%~dp0backend-next"

echo.
echo ╔═══════════════════════════════════╗
echo ║     MindFlow - 智能专注助手      ║
echo ╚═══════════════════════════════════╝
echo.
echo [1] 浏览器模式
echo     命令行启动，浏览器打开 Dashboard
echo.
echo [2] 仅启动后端
echo     只启动 API 服务，不打开界面
echo.
echo 注：旧版 backend/ 的系统托盘模式已随其删除而移除
echo     （backend-next 暂无等价托盘入口，需另行补做）。
echo.
set /p choice="请选择 (1/2): "

if "%choice%"=="1" goto browser
if "%choice%"=="2" goto backend
echo 无效选择，退出
pause
exit /b

:browser
echo.
echo 正在启动后端...
start "" /B python -m mindflow.main
echo 等待后端就绪...
timeout /t 3 /nobreak > nul
echo 正在打开 Dashboard...
start http://127.0.0.1:8765/docs
echo.
echo 后端已启动，关闭此窗口不会停止后端
echo 需要停止时请关闭 python 窗口
goto end

:backend
echo.
echo 正在启动后端...
python -m mindflow.main
goto end

:end
pause
