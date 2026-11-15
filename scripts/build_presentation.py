"""Light presentation figures from every window of the actual production atlas.

python3.12 scripts/build_presentation.py --api ../api --cache ../reports/presentation-cache
Dependencies: numpy, pandas, scipy, matplotlib. All searches use production code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation'
FIG = OUT / 'figures'
DATA = OUT / 'data'
BLUE, RED, INK, MUTED, RULE = '#006fee', '#c84045', '#18181b', '#71717a', '#e4e4e7'
COLORS = {'earth': BLUE, 'moon': '#71717a', 'mars': '#c66b42'}
NAMES = {'earth': 'Земля', 'moon': 'Луна', 'mars': 'Марс'}
FEATURES = ['Перепад высот', 'Наклон плоскости', 'Неровность']
FEATURE_RU = {'profile': 'Форма рельефа', 'range': 'Перепад высот', 'slope': 'Наклон плоскости', 'roughness': 'Неровность'}
COUNTRY_RU = {'Iran': 'Иран', 'Chile': 'Чили', 'Afghanistan': 'Афганистан', 'Niger': 'Нигер', 'United States of America': 'США'}
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
if FONT_PATH.exists():
    fontManager.addfont(str(FONT_PATH))
    FONT = FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT = 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT, 'font.size': 13, 'text.color': INK,
    'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': RULE, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'medium', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.unicode_minus': True})


def num(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def score(value):
    return f'{value:.4g}'.replace('.', ',')


def integer(value):
    return f'{value:,}'.replace(',', ' ')


def frame(kicker, title, subtitle='', number=''):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .942, kicker.upper(), fontsize=11, color=BLUE, weight='bold')
    fig.text(.055, .872, title, fontsize=29, weight='bold')
    if subtitle:
        fig.text(.055, .82, subtitle, fontsize=13, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .945], [.792, .792], transform=fig.transFigure, color=RULE, lw=.8))
    fig.text(.945, .942, number, ha='right', fontsize=11, color=MUTED)
    return fig


def footer(fig, conclusion, source):
    fig.add_artist(plt.Line2D([.055, .945], [.151, .151], transform=fig.transFigure, color=RULE, lw=.8))
    fig.text(.055, .102, conclusion, fontsize=15, weight='medium')
    fig.text(.055, .045, source, fontsize=9.5, color=MUTED)


def clean_axis(ax):
    ax.tick_params(length=0, pad=8, labelsize=11)
    ax.grid(axis='y', color=RULE, lw=.65, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, pdf=None):
    fig.savefig(FIG / f'{name}.png', dpi=200)
    fig.savefig(FIG / f'{name}.svg')
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def actual_examples(api):
    sys.path.insert(0, str(api / 'app'))
    from terrain_explain import explain_candidates
    from countries import country_at
    weights = {'slope': .45, 'roughness': .25, 'profile': .20, 'range': .10}
    queries = [dict(planet='mars', lat=18.4447, lon=77.4508, weights=weights),
               dict(planet='moon', lat=.67416, lon=23.47314, weights=weights)]
    result = subprocess.run(['node', 'app/terrain_engine.cjs'], cwd=api,
        input=''.join(json.dumps(q) + '\n' for q in queries), text=True,
        capture_output=True, timeout=90, check=True)
    examples = []
    for query, line in zip(queries, result.stdout.splitlines(), strict=True):
        response = json.loads(line)
        assert response['kind'] == 'results'
        response['weights'] = weights
        response['request'] = query
        response['matches'] = [{**m, **country_at(m['lat'], m['lon'])}
            for m in explain_candidates(response['matches'], response['criteria'], weights)]
        for m in response['matches']:
            e = m['explanation']
            assert abs(e['baseline_score'] + sum(f['shap_value'] for f in e['features']) - m['score']) < 1e-10
        examples.append(response)
    return examples


def prepare(cache, api):
    metadata = json.loads((cache / 'metadata.json').read_text())
    stats, summaries, correlations, quantiles = {}, [], {}, {}
    probabilities = np.linspace(0, 1, 501)
    for body, item in metadata['bodies'].items():
        data = np.memmap(cache / f'{body}.f64', mode='r', dtype='<f8', shape=tuple(item['shape']))
        assert np.isfinite(data).all() and (data >= 0).all()
        stats[body] = data
        quantiles[body] = np.quantile(data, probabilities, axis=0)
        correlations[body] = spearmanr(data, axis=0).statistic
        pd.DataFrame(correlations[body], index=metadata['features'], columns=metadata['features']).to_csv(DATA / f'{body}_spearman.csv')
        pd.DataFrame({'fraction': probabilities, **{name: quantiles[body][:, j] for j, name in enumerate(metadata['features'])}}).to_csv(DATA / f'{body}_ecdf_quantiles.csv', index=False)
        for j, feature in enumerate(metadata['features']):
            q = np.quantile(data[:, j], [.1, .5, .9, .95, .99])
            summaries.append(dict(body=body, feature=feature, windows=len(data),
                zero_fraction=float(np.mean(data[:, j] == 0)), minimum=float(data[:, j].min()),
                p10=q[0], median=q[1], p90=q[2], p95=q[3], p99=q[4], maximum=float(data[:, j].max())))
    pd.DataFrame(summaries).to_csv(DATA / 'production_feature_summary.csv', index=False)
    examples = actual_examples(api)
    (DATA / 'actual_search_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n')
    (DATA / 'atlas_provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    # Choose a real candidate with both signs to make the explanatory example readable.
    matches = examples[0]['matches']
    chosen = next((i for i, m in enumerate(matches) if
        any(f['shap_value'] > 1e-8 for f in m['explanation']['features']) and
        any(f['shap_value'] < -1e-8 for f in m['explanation']['features'])), 0)
    return metadata, pd.DataFrame(summaries), correlations, quantiles, examples[0], chosen


def source_figure(meta, pdf):
    fig = frame('Данные / происхождение', 'Три мира. Одинаковый формат сравнения.',
        'Цифровые модели рельефа → окна 10 × 10 км → 25 отсчётов высоты', '01 / 08')
    x_positions = [.07, .38, .69]
    sources = {'moon': ('NASA · LRO LOLA', 'LDEM_64 · ≈474 м'),
        'mars': ('NASA · MGS MOLA', 'MEGDR 32 · ≈1853 м'),
        'earth': ('NOAA · ETOPO 2022', '60″ · ≈1853 м')}
    for x, body in zip(x_positions, ['moon', 'mars', 'earth']):
        info = meta['bodies'][body]
        fig.text(x, .705, NAMES[body], fontsize=21, weight='bold', color=COLORS[body])
        fig.text(x, .647, sources[body][0], fontsize=15)
        fig.text(x, .603, sources[body][1], fontsize=12, color=MUTED)
        fig.text(x, .495, integer(info['windows']), fontsize=34, weight='bold')
        fig.text(x, .45, 'окон в полном EDA', fontsize=12, color=MUTED)
        if body == 'earth':
            fig.text(x, .39, 'после региональной маски воды', fontsize=11, color=MUTED)
        ax = fig.add_axes([x, .25, .095, .12])
        ax.imshow(np.indices((5, 5)).sum(axis=0) % 2, cmap=LinearSegmentedColormap.from_list('samples', ['#f4f4f5', '#d4d4d8']), vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.text(x + .115, .29, '5 × 5\nшаг 2,5 км', fontsize=13, color=MUTED, linespacing=1.6)
    footer(fig, 'Сравниваем форму поверхности на общем масштабе измерений.',
        'Источники: NASA PDS / LOLA / MOLA; NOAA NCEI ETOPO 2022. Нативное разрешение указано приблизительно у экватора.\nТемпература и атмосфера: NASA POWER, MCD 6.2, Diviner — отдельный контекст, вне рейтинга.')
    save(fig, '01_data_sources', pdf)


def distribution_figure(meta, summary, quantiles, pdf):
    total = sum(x['windows'] for x in meta['bodies'].values())
    fig = frame('EDA / все окна атласа', 'Наклон, перепад и неровность различаются между телами.',
        f'{integer(total)} окон; каждая линия рассчитана по всем допустимым окнам соответствующего тела', '02 / 08')
    axes = fig.subplots(1, 3, gridspec_kw={'left': .07, 'right': .95, 'bottom': .285, 'top': .69, 'wspace': .27})
    linthresh = [100, .1, 10]
    ticks = [[0, 100, 1000, 10000], [0, .1, 1, 10], [0, 10, 100, 1000]]
    for j, ax in enumerate(axes):
        for body in ['earth', 'moon', 'mars']:
            x = quantiles[body][:, j]
            ax.plot(x, np.linspace(0, 100, len(x)), color=COLORS[body], lw=2.5, label=NAMES[body])
        ax.set_xscale('symlog', linthresh=linthresh[j]); ax.set_xticks(ticks[j]); ax.set_xticklabels([str(v).replace('.', ',') for v in ticks[j]])
        ax.set_xlim(0, max(quantiles[b][-2, j] for b in quantiles) * 1.12)
        ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); clean_axis(ax)
        ax.set_title(FEATURES[j], loc='left', fontsize=16, pad=20)
        ax.set_xlabel('градусы' if j == 1 else 'метры', labelpad=10)
        if j == 0: ax.set_ylabel('Доля окон с величиной ≤ x, %')
        if j == 2: ax.legend(frameon=False, loc='lower right', fontsize=12)
    text = '   ·   '.join(f'{NAMES[b]}: {num(summary.query("body == @b and feature == \'range_m\'").iloc[0]["median"], 0)} м' for b in ['earth', 'moon', 'mars'])
    fig.text(.07, .208, 'Медианный перепад высот: ' + text, fontsize=13)
    footer(fig, 'Одного признака недостаточно: сравниваем наклон, перепад и неровность вместе.',
        'Полный production-атлас; доли окон, не площади планет. Ось x сжата логарифмически после линейного участка.\nКрайний хвост за 99,8-м процентилем не показан; исходные min/max и квантили приложены в CSV.')
    save(fig, '02_global_terrain_eda', pdf)


def shap_bars(ax, candidate, compact=False):
    features = sorted(candidate['explanation']['features'], key=lambda f: abs(f['shap_value']), reverse=True)
    values = [f['shap_value'] for f in features]
    ax.barh(range(4), values, color=[BLUE if v < 0 else RED for v in values], height=.48)
    ax.set_yticks(range(4), [FEATURE_RU[f['feature']] for f in features], fontsize=10 if compact else 12)
    ax.invert_yaxis(); ax.axvline(0, color='#a1a1aa', lw=.8)
    extent = max(abs(v) for v in values) or 1
    ax.set_xlim(-extent * 1.8, extent * 2.4)
    for i, v in enumerate(values):
        ax.text(.99, i, ('+' if v > 0 else '') + num(v * 1000, 3), transform=ax.get_yaxis_transform(),
            va='center', ha='right', fontsize=9 if compact else 11, color=BLUE if v < 0 else RED)
    ax.set_xticks([]); ax.tick_params(axis='y', length=0, pad=9)
    for spine in ax.spines.values(): spine.set_visible(False)


def relationship_figure(correlations, example, chosen, pdf):
    fig = frame('EDA → объяснимый метод', 'От связей в данных — к объяснению рекомендации.',
        'Слева — статистика всех окон; справа — локальное объяснение одного реального кандидата', '03 / 08')
    cmap = LinearSegmentedColormap.from_list('correlation', ['#fafafa', '#d7e8fd', BLUE])
    labels = ['Перепад', 'Наклон', 'Неровность']
    for x, body in [(.08, 'earth'), (.365, 'mars')]:
        ax = fig.add_axes([x, .31, .225, .36])
        ax.imshow(correlations[body], cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(3), labels, rotation=25, ha='right', fontsize=11)
        ax.set_yticks(range(3), labels, fontsize=11); ax.tick_params(length=0)
        ax.set_title(f'{NAMES[body]} · Spearman ρ', fontsize=15, loc='left', pad=18)
        for i in range(3):
            for j in range(3):
                v = correlations[body][i, j]
                ax.text(j, i, num(v), ha='center', va='center', fontsize=15, color='white' if v > .65 else INK)
        for spine in ax.spines.values(): spine.set_visible(False)
    ax = fig.add_axes([.76, .355, .17, .28])
    shap_bars(ax, example['matches'][chosen], True)
    fig.text(.705, .715, 'Пример Exact SHAP', fontsize=15, weight='bold')
    fig.text(.705, .67, f'Jezero → кандидат №{chosen + 1:02d}', fontsize=11, color=MUTED)
    fig.text(.705, .638, r'Вклад в балл $\times10^{-3}$', fontsize=10, color=MUTED)
    fig.text(.71, .265, 'Синий: уменьшает несовпадение\nКрасный: увеличивает', fontsize=11, color=MUTED, linespacing=1.5)
    footer(fig, 'SHAP показывает вклад каждого критерия в конкретную рекомендацию.',
        'Корреляции: полный production-атлас. SHAP: четыре слагаемых действующего score; фон — среднее десяти рекомендаций.\nКорреляция описывает данные, SHAP объясняет вычисленный балл. Это разные задачи; ни одна диаграмма не доказывает причинность.')
    save(fig, '03_eda_to_explainability', pdf)


def objective_figure(pdf):
    fig = frame('Метод / приоритеты', 'Задача испытаний задаёт веса критериев.',
        'Цель исследования меняет веса четырёх критериев, сохраняя общую формулу сравнения', '04 / 08')
    ids = ['profile', 'range', 'slope', 'roughness']
    orders = [['slope', 'roughness', 'profile', 'range'], ['slope', 'range', 'profile', 'roughness'],
        ['profile', 'range', 'slope', 'roughness'], ['slope', 'roughness', 'range', 'profile'], ['profile', 'range', 'roughness', 'slope']]
    table = np.array([[dict(zip(order, [.45, .25, .2, .1]))[name] for name in ids] for order in orders])
    ax = fig.add_axes([.27, .29, .64, .42])
    ax.imshow(table, cmap=LinearSegmentedColormap.from_list('weights', ['#f4f7fb', BLUE]), vmin=.1, vmax=.45, aspect='auto')
    ax.set_xticks(range(4), [FEATURE_RU[k] for k in ids], fontsize=13)
    ax.set_yticks(range(5), ['Ровер', 'Подготовка астронавтов', 'Посадочные сенсоры', 'Земляные работы', 'Геологическая практика'], fontsize=13)
    ax.tick_params(length=0, pad=15)
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f'{table[i,j]:.0%}', ha='center', va='center', color='white' if table[i,j] >= .4 else INK, fontsize=18)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.27, .2, 'Нормировки различий: профиль 100 м  ·  перепад 150 м  ·  наклон 2°  ·  неровность 50 м', fontsize=12, color=MUTED)
    footer(fig, 'Ранжируем кандидатов под задачу пользователя.',
        'Действующий поиск: взвешенная сумма квадратов нормированных различий. Меньший балл — ближе рельеф.\nВеса и нормировки заданы разработчиками; они не являются обученными вероятностями или пределами безопасности.')
    save(fig, '04_objective_weights', pdf)


def match_figure(example, chosen, pdf):
    m = example['matches'][chosen]
    fig = frame('Метод / реальный пример', 'Сравниваем форму, сохраняя физический масштаб.',
        f'Jezero, Марс → {COUNTRY_RU.get(m["country"], m["country"])}, кандидат №{chosen + 1:02d} · задача: испытания ровера', '05 / 08')
    source = np.array(example['query'], dtype=float).reshape(5, 5)
    earth = np.array(m['values'], dtype=float).reshape(5, 5)
    source -= source.mean(); earth -= earth.mean()
    x, y = np.meshgrid(np.linspace(-5, 5, 5), np.linspace(5, -5, 5))
    limit = max(abs(source).max(), abs(earth).max())
    cmap = LinearSegmentedColormap.from_list('terrain', ['#eff6ff', '#9fc5fb', '#0065d9'])
    for pos, values, title in [(121, source, 'Марс · окно назначения'), (122, earth, 'Земля · профиль после совмещения')]:
        ax = fig.add_subplot(pos, projection='3d')
        ax.plot_surface(x, y, values, cmap=cmap, vmin=-limit, vmax=limit, edgecolor='#ffffff80', linewidth=.5, antialiased=True)
        ax.scatter(x, y, values, c=INK, s=8, alpha=.5)
        ax.set_zlim(-limit, limit); ax.view_init(elev=26, azim=-130)
        ax.set_xlabel('км', fontsize=11, labelpad=4); ax.set_ylabel('км', fontsize=11, labelpad=4); ax.set_zlabel('высота, м', fontsize=11, labelpad=6)
        ax.set_title(title, fontsize=15, pad=4)
        ax.tick_params(labelsize=9, pad=0)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]: axis.pane.fill = False
    fig.subplots_adjust(left=.035, right=.955, bottom=.25, top=.72, wspace=.05)
    fig.text(.5, .205, f'RMSE профиля: {num(m["error"])} м  ·  25 пар высот  ·  собственная средняя высота удалена', ha='center', fontsize=14)
    footer(fig, 'Сходство измеряется по данным рельефа, а затем объясняется по критериям.',
        f'Источник: production-поиск. Земной центр: {m["lat"]:.4f}°, {m["lon"]:.4f}°. Общий шаг отсчётов 2,5 км.\nЛинии соединяют 25 отсчётов; детали между ними не измерены. Вертикальная шкала визуально увеличена; единицы подписаны.')
    save(fig, '05_real_terrain_match', pdf)


def ranking_figure(example, chosen, pdf):
    fig = frame('Метод / результат', 'Десять земных кандидатов для выбранной задачи.',
        'Jezero · испытания ровера · балл несовпадения D: меньше — лучше', '06 / 08')
    matches = example['matches']
    ax = fig.add_axes([.31, .24, .6, .48])
    values = [m['score'] for m in matches]
    ax.barh(range(10), values, height=.56, color=[BLUE if i == chosen else '#d4d4d8' for i in range(10)])
    ax.set_yticks(range(10), [f'{i+1:02d}  {COUNTRY_RU.get(m["country"],m["country"])} · {m["lat"]:.1f}°, {m["lon"]:.1f}°' for i, m in enumerate(matches)], fontsize=11)
    ax.invert_yaxis(); ax.set_xlabel('Баллы D, безразмерная величина', fontsize=12, labelpad=12)
    ax.grid(axis='x', color=RULE, lw=.6); ax.set_axisbelow(True); ax.tick_params(length=0, pad=9)
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
    ax.set_xlim(0, max(values) * 1.18)
    for i, value in enumerate(values): ax.text(value + max(values)*.012, i, score(value), va='center', fontsize=11, color=BLUE if i == chosen else MUTED)
    footer(fig, 'Пользователь сравнивает альтернативы и открывает объяснение выбранного места.',
        'Реальный ответ API. Список учитывает географическое правило: до трёх мест США при достаточном числе кандидатов.\nБаллы не являются процентами сходства. Синий — кандидат, подробно показанный на соседних слайдах.')
    save(fig, '06_candidate_ranking', pdf)


def shap_figure(example, chosen, pdf):
    m = example['matches'][chosen]
    exp = m['explanation']
    features = sorted(exp['features'], key=lambda f: abs(f['shap_value']), reverse=True)
    fig = frame('Объяснимость / Exact SHAP', 'Каждый вклад в итоговый балл можно проверить.',
        f'Jezero → {COUNTRY_RU.get(m["country"], m["country"])}, кандидат №{chosen+1:02d} · фон: средний балл этих десяти кандидатов', '07 / 08')
    ax = fig.add_axes([.25, .31, .63, .40])
    cumulative = exp['baseline_score']
    ax.barh(0, cumulative, color='#a1a1aa', height=.58)
    endpoints = [cumulative]
    for i, f in enumerate(features, 1):
        value = f['shap_value']; end = cumulative + value
        ax.barh(i, abs(value), left=min(cumulative, end), color=BLUE if value < 0 else RED, height=.58)
        ax.plot([cumulative, cumulative], [i-1+.3, i-.3], color='#a1a1aa', lw=.7, ls=':')
        ax.text(max(cumulative, end) + exp['baseline_score']*.025, i, ('+' if value > 0 else '') + score(value), va='center', fontsize=12, color=BLUE if value < 0 else RED)
        cumulative = end; endpoints.append(cumulative)
    assert abs(cumulative - exp['score']) < 1e-10
    ax.barh(5, exp['score'], color=INK, height=.58)
    for i, value in [(0, exp['baseline_score']), (5, exp['score'])]:
        ax.text(value + exp['baseline_score']*.025, i, score(value), va='center', fontsize=12, weight='bold')
    labels = ['Средний балл списка'] + [FEATURE_RU[f['feature']] for f in features] + ['Балл кандидата']
    ax.set_yticks(range(6), labels, fontsize=13); ax.invert_yaxis()
    ax.set_xlim(0, max(endpoints) * 1.4); ax.set_xlabel('Баллы несовпадения D', fontsize=12, labelpad=12)
    ax.tick_params(length=0, pad=12); ax.grid(axis='x', color=RULE, lw=.6); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    fig.text(.25, .196, f'{score(exp["baseline_score"])}  +  ({score(exp["score"]-exp["baseline_score"])})  =  {score(exp["score"])}', fontsize=16, weight='bold')
    footer(fig, 'Видно, какие характеристики улучшают совпадение и где остаётся различие.',
        'Синий вклад уменьшает D, красный увеличивает. Exact interventional SHAP аддитивного score по четырём дескрипторам.\nЭто объяснение относительно текущего списка, не уверенность модели и не причинное заключение. Значения округлены.')
    save(fig, '07_exact_shap', pdf)
    compact = plt.figure(figsize=(6.4, 3.2))
    ax = compact.add_axes([.3, .2, .66, .67]); shap_bars(ax, m, True)
    compact.text(.03, .94, r'Exact SHAP · вклад в балл $\times10^{-3}$', fontsize=13, weight='bold')
    compact.text(.03, .04, '← меньше несовпадение       больше →', fontsize=10, color=MUTED)
    save(compact, '09_shap_inset')


def experimental_figure(pdf):
    catalog = json.loads((ROOT / 'artifacts/mars/mars_types.json').read_text())
    fig = frame('Приложение / обученный исследовательский прототип', 'MiniBatchKMeans: сравнение 6, 10 и 14 групп.',
        '150 000 наблюдений Марса · log1p + RobustScaler · seed 42 · отдельный эксперимент', '08 / 08')
    ax = fig.add_axes([.08, .31, .32, .38])
    diagnostics = catalog['diagnostics']
    ks = [d['k'] for d in diagnostics]; sil = [d['silhouette'] for d in diagnostics]
    ax.plot(ks, sil, color='#a1a1aa', lw=1.5)
    ax.scatter(ks, sil, c=[BLUE if k == 10 else INK for k in ks], s=70, zorder=3)
    for k, s in zip(ks, sil): ax.annotate(num(s, 3), (k, s), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=14)
    ax.set_ylim(.2, .47); ax.set_xticks(ks); ax.set_xlabel('Количество кластеров k', labelpad=12); ax.set_ylabel('Silhouette · выше — лучше', labelpad=12); clean_axis(ax)
    fig.text(.08, .745, 'Диагностика разделимости', fontsize=15, weight='bold')
    features = np.array([[t['features_median'][k] for k in catalog['fields']] for t in catalog['types']])
    logged = np.log1p(features); normalized = (logged-logged.min(axis=0))/(logged.max(axis=0)-logged.min(axis=0))
    ax = fig.add_axes([.58, .245, .32, .48])
    ax.imshow(normalized, cmap=LinearSegmentedColormap.from_list('types', ['#f4f7fb', BLUE]), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(3), ['Медиана\nуклона, °', 'P90\nуклона, °', 'Перепад\np95−p05, м'], fontsize=11)
    ax.set_yticks(range(10), [f'{i+1:02d}' for i in range(10)], fontsize=11); ax.tick_params(length=0, pad=10)
    for i in range(10):
        for j in range(3):
            ax.text(j, i, num(features[i,j], 2 if j<2 else 0), ha='center', va='center', fontsize=11, color='white' if normalized[i,j]>.65 else INK)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.58, .745, 'Медианные характеристики 10 групп', fontsize=15, weight='bold')
    footer(fig, 'Десять групп выбраны для удобства демо; максимальный silhouette здесь у k = 6.',
        'Исследовательская модель не используется текущим /terrain/search. Признаки охватывают окрестности порядка 50 км.\nSilhouette рассчитан на 2000 точках обучающей выборки, не на независимом полевом тесте. Цвет таблицы нормирован по колонке.')
    save(fig, '08_experimental_ml', pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', type=Path, default=ROOT.parent/'api')
    parser.add_argument('--cache', type=Path, default=ROOT.parent/'reports/presentation-cache')
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(exist_ok=True)
    meta, summary, corrs, quantiles, example, chosen = prepare(args.cache.resolve(), args.api.resolve())
    with PdfPages(OUT/'project-slides-light.pdf') as pdf:
        source_figure(meta, pdf); distribution_figure(meta, summary, quantiles, pdf)
        relationship_figure(corrs, example, chosen, pdf); objective_figure(pdf)
        match_figure(example, chosen, pdf); ranking_figure(example, chosen, pdf)
        shap_figure(example, chosen, pdf); experimental_figure(pdf)
    (DATA/'selected_example.json').write_text(json.dumps({'body': 'mars', 'location': 'Jezero', 'candidate_index': chosen,
        'candidate': example['matches'][chosen], 'selection_note': 'First actual shortlist candidate with both positive and negative SHAP contributions, chosen for explanation readability.'}, ensure_ascii=False, indent=2)+'\n')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix != '.zip' and p.name != 'manifest.json')
    manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    with zipfile.ZipFile(OUT/'presentation-figures-light.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.zip': archive.write(p, p.relative_to(OUT))
    print(json.dumps({'all_windows': sum(b['windows'] for b in meta['bodies'].values()),
        'selected_candidate': chosen+1, 'country': example['matches'][chosen]['country'],
        'output': str(OUT), 'summary': summary.to_dict(orient='records')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
