@echo off
REM Instala o servidor MCP do projeto no Claude Desktop (Windows). Dois cliques neste arquivo.
REM Log completo em setup_mcp.log. Documentacao: README, secao "Uso via MCP (Claude Desktop)".
cd /d "%~dp0"
echo ==== %date% %time% ==== > setup_mcp.log
set PY=
where py >nul 2>&1 && set PY=py -3
if not defined PY where python >nul 2>&1 && set PY=python
if not defined PY (
  echo [ERRO] Python nao encontrado >> setup_mcp.log
  echo Python nao encontrado. Instale o Python 3.11+ ^(python.org^) e rode de novo.
  pause & exit /b 1
)
echo [1/3] Criando o ambiente .venv com %PY% ...
if not exist .venv\Scripts\python.exe %PY% -m venv .venv >> setup_mcp.log 2>&1
if not exist .venv\Scripts\python.exe (
  echo [ERRO] Nao consegui criar o .venv. Veja setup_mcp.log
  pause & exit /b 1
)
echo [2/3] Instalando dependencias ^(alguns minutos^) ...
.venv\Scripts\python.exe -m pip install --upgrade pip >> setup_mcp.log 2>&1
.venv\Scripts\python.exe -m pip install -r requirements-tool.txt >> setup_mcp.log 2>&1
if errorlevel 1 (
  echo [ERRO] pip install falhou. Veja setup_mcp.log
  pause & exit /b 1
)
echo [3/3] Testando o modelo e registrando no Claude Desktop ...
.venv\Scripts\python.exe scripts\setup\configure_claude_desktop.py >> setup_mcp.log 2>&1
if errorlevel 1 (
  echo [ERRO] O modelo nao carregou neste computador ^(provavel bloqueio de DLL do CatBoost/pandas^). Nada foi gravado no Claude Desktop. Veja setup_mcp.log
  pause & exit /b 1
)
echo.
echo PRONTO. Feche o Claude Desktop por completo ^(inclusive da bandeja^) e abra de novo.
echo PRONTO >> setup_mcp.log
pause
