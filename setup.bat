@echo off
REM Windows setup: install deps + git hooks.
REM Run: setup.bat

set ROOT=%~dp0
set SRC=%ROOT%src

echo ^>^>^> uv sync
uv sync --directory "%SRC%"
if errorlevel 1 exit /b 1

echo ^>^>^> pre-commit install
uv run --directory "%SRC%" pre-commit install
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Git hooks are active.
