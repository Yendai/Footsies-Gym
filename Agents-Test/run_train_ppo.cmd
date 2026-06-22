@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

pushd "%REPO_ROOT%"
"%PYTHON_CMD%" -u "Agents-Test\train_ppo.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
