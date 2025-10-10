@echo off
REM Pre-push check script - Run all quality checks before pushing to remote
REM Note: For colored output on Windows, use Git Bash and run ./scripts/check.sh instead
setlocal enabledelayedexpansion

echo ==================================================
echo Running pre-push checks...
echo ==================================================

set FAILED=0

REM Check 1: Ruff linting
echo.
echo [1/3] Running Ruff linting...
uv run ruff check .
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Ruff passed
) else (
    echo [FAIL] Ruff failed
    set FAILED=1
)

REM Check 2: MyPy type checking
echo.
echo [2/3] Running MyPy type checking...
uv run mypy src/maktaba --no-error-summary 2>&1 | findstr /C:"error:" > mypy_errors.tmp
if exist mypy_errors.tmp (
    for /f %%A in ('type mypy_errors.tmp ^| find /c /v ""') do set ERROR_COUNT=%%A
    del mypy_errors.tmp
) else (
    set ERROR_COUNT=0
)

echo MyPy found !ERROR_COUNT! errors

if !ERROR_COUNT! GTR 20 (
    echo [FAIL] MyPy failed (threshold: 20 errors^)
    set FAILED=1
) else (
    echo [PASS] MyPy passed (!ERROR_COUNT! errors, threshold: 20^)
)

REM Check 3: Pytest
echo.
echo [3/3] Running tests...
uv run python -m pytest tests/ -v
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Tests passed
) else (
    echo [FAIL] Tests failed
    set FAILED=1
)

REM Summary
echo.
echo ==================================================
if !FAILED! EQU 0 (
    echo [SUCCESS] All checks passed! Safe to push.
    echo ==================================================
    exit /b 0
) else (
    echo [ERROR] Some checks failed. Please fix before pushing.
    echo ==================================================
    exit /b 1
)
