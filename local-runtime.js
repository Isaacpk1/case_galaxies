/* Adaptador mínimo para executar localmente o painel criado para Dashboard. */
(() => {
  const names = ['base', 'modelos_teste', 'evolucao', 'ordenacao', 'recortes',
    'canais', 'maiores_erros', 'intervalo', 'clusters', 'importancia', 'atributos', 'e17'];
  const values = new Map(names.map((name) => [name, { status: 'loading', data: [] }]));
  const calculations = new Map();
  let listener = null;

  async function reload() {
    const status = document.getElementById('local-data-status');
    if (status) status.textContent = 'Carregando dados locais…';
    await Promise.all(names.map(async (name) => {
      try {
        const base = window.GX_DATA_BASE || '/api/datasets/';
        const response = await fetch(`${base}${name}${window.GX_STATIC_SITE ? '.json' : ''}`, { cache: 'no-store' });
        const rows = await response.json();
        if (!response.ok) throw new Error(rows.error || `HTTP ${response.status}`);
        values.set(name, { status: 'ok', data: rows });
      } catch (error) {
        values.set(name, { status: 'error', data: [], error: String(error) });
      }
    }));
    for (const [name, calc] of calculations) {
      try {
        const inputs = calc.inputs.map((id) => {
          const source = values.get(id);
          if (!source || source.status !== 'ok') throw new Error(`Fonte indisponível: ${id}`);
          return source.data;
        });
        values.set(name, { status: 'ok', data: calc.fn(...inputs) });
      } catch (error) {
        values.set(name, { status: 'error', data: [], error: String(error) });
      }
    }
    const failed = names.filter((name) => values.get(name).status !== 'ok');
    if (status) status.textContent = failed.length
      ? `Dados locais: ${failed.length} fonte(s) indisponível(is) — ${failed.join(', ')}`
      : (window.GX_STATIC_SITE ? 'Dados publicados carregados' : 'Dados locais atualizados');
    if (listener && window.d3) listener();
    else if (status && !window.d3) status.textContent += ' · biblioteca de gráficos indisponível';
  }

  window.dash = {
    data: (name) => values.get(name) || { status: 'loading', data: [] },
    calc: (name, definition) => calculations.set(name, definition),
    onData: (callback) => { listener = callback; reload(); },
    setLink: (name) => { window.location.hash = name; },
  };
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('local-data-refresh')?.addEventListener('click', reload);
  });
})();
