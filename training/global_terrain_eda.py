"""Explain every global EDA figure using computed results and explicit populations."""
from __future__ import annotations

import base64
import html
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir()+"/analog-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from global_terrain_data import DATA, ROOT
from global_terrain_extract import OUT, STEP, weighted_quantile
from terrain_eda import FEATURES

COLORS = {"Earth": "#267da6", "Mars": "#c65d3b"}
SHORT = ["Median slope", "Slope p90", "Fraction >5 deg", "Relative relief", "Plane residual SD"]


def weighted_rank(values, weights):
    """Weighted empirical mid-CDF; ties receive the same rank."""
    _, inverse = np.unique(values, return_inverse=True)
    group_weights = np.bincount(inverse, weights=weights)
    ranks = (np.cumsum(group_weights)-.5*group_weights)/group_weights.sum()
    return ranks[inverse]


def weighted_spearman(values, weights):
    ranks = np.column_stack([weighted_rank(values[:, i], weights) for i in range(values.shape[1])])
    means = np.average(ranks, weights=weights, axis=0)
    centered = ranks-means
    cov = (centered*weights[:, None]).T@centered/weights.sum()
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide='ignore', invalid='ignore'):
        return cov/(sd[:, None]*sd[None, :])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples, patches, coverage, stats = {}, {}, {}, {}
    for body in ("Earth", "Mars"):
        samples[body] = dict(np.load(DATA / f"{body.lower()}_cells.npz"))
        patches[body] = pd.read_csv(OUT / f"{body.lower()}_patches.csv")
        coverage[body] = pd.read_csv(OUT / f"{body.lower()}_coverage.csv")
        s, p, c = samples[body], patches[body], coverage[body]
        weights = s["area_km2"].astype(float)
        stats[body] = {
            "cells": len(weights), "patches": len(p), "tiles_processed": len(c),
            "tiles_with_slope_data": int((c.valid_slope_cells > 0).sum()),
            "domain_area_million_km2": float(c.domain_area_km2.sum()/1e6),
            "elevation_coverage_pct": float(c.valid_elevation_area_km2.sum()/c.domain_area_km2.sum()*100),
            "slope_coverage_pct": float(c.valid_slope_area_km2.sum()/c.domain_area_km2.sum()*100),
            "slope_area_million_km2": float(weights.sum()/1e6),
            "slope_median_deg": weighted_quantile(s["slope_deg"], weights, .5),
            "slope_p90_deg": weighted_quantile(s["slope_deg"], weights, .9),
            "slope_gt5_area_pct": float(np.average(s["slope_deg"] > 5, weights=weights)*100),
            "patch_area_million_km2": float(p.area_km2.sum()/1e6),
        }
        for feature in FEATURES:
            stats[body]["patch_median__"+feature] = weighted_quantile(p[feature].to_numpy(), p.area_km2.to_numpy(), .5)
    pd.DataFrame(stats).T.to_csv(OUT / "global_summary.csv")
    plt.rcParams.update({"figure.facecolor": "#f4f7fb", "axes.facecolor": "white",
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 150})
    explanations = []

    def save(fig, name, title, what, finding, limit):
        fig.savefig(OUT / name, bbox_inches="tight")
        plt.close(fig)
        explanations.append({"file": name, "title": title, "what": what, "finding": finding, "limit": limit})

    def tile_map(frame, field):
        image = np.full((18, 36), np.nan)
        for row in frame.itertuples():
            image[(int(row.south)+90)//10, (int(row.west)+180)//10] = getattr(row, field)
        return image

    def map_axes(ax, image, **kwargs):
        im = ax.imshow(image, origin="lower", extent=(-180, 180, -90, 90), aspect="auto", **kwargs)
        ax.set_xlabel("Longitude (deg east)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        return im

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    for ax, body in zip(axes, ("Earth", "Mars")):
        c = coverage[body].copy()
        c["coverage_pct"] = c.valid_slope_area_km2/c.domain_area_km2.replace(0, np.nan)*100
        im = map_axes(ax, tile_map(c, "coverage_pct"), cmap="viridis", vmin=0, vmax=100)
        ax.set_title(f"{body}: slope coverage within eligible domain")
    fig.colorbar(im, ax=axes.tolist(), shrink=.8, label="Valid slope area / domain area (%)")
    save(fig, "01_global_coverage.png", "1. Покрытие: где вообще есть данные для выводов",
         "Каждый прямоугольник — географический блок 10° × 10°. Цвет — доля площади с вычисленным уклоном. "
         "Серые/пустые области Земли без суши не входят в знаменатель. Для Марса знаменатель — вся поверхность.",
         f"Обработаны все 648 блоков каждой планеты. Валидные уклоны покрывают "
         f"{stats['Earth']['slope_coverage_pct']:.2f}% маски земной суши и "
         f"{stats['Mars']['slope_coverage_pct']:.2f}% марсианской поверхности.",
         "Потери у побережий связаны с маской и необходимостью соседних высот для градиента. "
         "Заполненная ячейка MOLA может быть интерполированной: покрытие не равно точности.")

    all_medians = np.concatenate([coverage[b].median_slope_deg.dropna().to_numpy() for b in stats])
    upper = max(float(np.quantile(all_medians, .99)), 1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    for ax, body in zip(axes, ("Earth", "Mars")):
        im = map_axes(ax, tile_map(coverage[body], "median_slope_deg"), cmap="magma", vmin=0, vmax=upper)
        ax.set_title(f"{body}: area-weighted median slope per 10-degree tile")
    fig.colorbar(im, ax=axes.tolist(), shrink=.8, extend="max", label="Median slope (deg on 10 km grid)")
    save(fig, "02_global_slope_maps.png", "2. Карты уклона: насколько рельеф неоднороден",
         "Цвет показывает медианный уклон в каждом блоке, с учётом площади ячеек. Шкала общая для двух планет; "
         "верхний 1% значений насыщается цветом. Более яркий цвет означает более крутой крупномасштабный рельеф.",
         "На карте видны географические различия, которые исчезли бы при усреднении целой планеты. "
         "Для поиска аналогов нужен профиль выбранного места, а не средний профиль Марса.",
         "Прямоугольники на двух планетах имеют разный физический размер. Здесь они служат только "
         "для отображения медианы уклона, вычисленного с одинаковым шагом 10 км; их перепады высот не сравниваются.")

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    max90 = max(stats[b]["slope_p90_deg"] for b in stats)
    for body in stats:
        s = samples[body]
        order = np.argsort(s["slope_deg"])
        vals, weights = s["slope_deg"][order], s["area_km2"][order].astype(float)
        ecdf = np.cumsum(weights)/weights.sum()
        ids = np.unique(np.r_[np.arange(0, len(vals), max(1, len(vals)//12000)), len(vals)-1])
        ax.plot(vals[ids], ecdf[ids], color=COLORS[body], label=body, linewidth=2)
        ax.axvline(stats[body]["slope_median_deg"], color=COLORS[body], linestyle=":", alpha=.6)
    ax.set_xlim(0, max(5, max90*2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Slope (deg on 10 km grid)")
    ax.set_ylabel("Fraction of valid surface area at or below slope")
    ax.set_title("Global area-weighted slope ECDF | Earth land vs Mars surface")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "03_global_slope_ecdf.png", "3. Распределение уклонов: какая доля площади ровная",
         "По горизонтали — уклон, по вертикали — доля исследованной площади с уклоном не больше заданного. "
         "Например, пересечение с 0.5 даёт медиану. Более высокая кривая при одном уклоне означает больше ровной площади.",
         f"Медианный уклон: суша Земли {stats['Earth']['slope_median_deg']:.3f}°, "
         f"Марс {stats['Mars']['slope_median_deg']:.3f}°. "
         f"Доля площади с уклоном >5°: {stats['Earth']['slope_gt5_area_pct']:.2f}% и "
         f"{stats['Mars']['slope_gt5_area_pct']:.2f}% соответственно. "
         "На этом масштабе и в этих моделях марсианское распределение сдвинуто к большим уклонам; "
         "при этом на обеих планетах преобладает сравнительно пологий крупный рельеф.",
         "Это уклоны после усреднения до 10 км, а не реальные уклоны под колесом ровера. "
         "Все доли считаются среди валидных площадей из графика 1; земные океаны не включены.")

    selected = [FEATURES[0], FEATURES[1], FEATURES[3], FEATURES[4]]
    labels = ["Median slope (deg)", "Slope p90 (deg)", "Relative relief p95-p05 (m)", "Plane residual SD (m)"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, feature, label in zip(axes.flat, selected, labels):
        cap = max(weighted_quantile(patches[b][feature].to_numpy(), patches[b].area_km2.to_numpy(), .995) for b in stats)
        bins = np.linspace(0, np.log1p(cap), 55)
        for body in stats:
            p = patches[body]
            mass, _ = np.histogram(np.log1p(p[feature]), bins=bins, weights=p.area_km2/p.area_km2.sum())
            ax.stairs(mass*100, bins, label=body, color=COLORS[body], linewidth=2)
        ticks = np.linspace(0, np.log1p(cap), 5)
        ax.set_xticks(ticks, [f"{v:.2g}" for v in np.expm1(ticks)])
        ax.set_xlabel(label+"; log1p spacing")
        ax.set_ylabel("Share of retained patch area per bin (%)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Global 100 x 100 km windows | common 10 km pixels | tails >99.5% omitted from display")
    save(fig, "04_global_patch_distributions.png", "4. Признаки окон: чем отличается ровный склон от неровного участка",
         "Каждое наблюдение — окно 100 × 100 км. Показаны медианный и высокий уклон, относительный перепад "
         "и SD отклонений от плоскости. Высота линии — доля площади в интервале, а не вероятность пригодности. "
         "Ось X сжата через log(1+x), подписи остаются в физических единицах.",
         f"Проанализированы {len(patches['Earth']):,} земных и {len(patches['Mars']):,} марсианских окон. "
         f"Медианный относительный перепад окна: {stats['Earth']['patch_median__relief_p95_p05_m']:.1f} м "
         f"на Земле и {stats['Mars']['patch_median__relief_p95_p05_m']:.1f} м на Марсе. "
         "Разные признаки нужны потому, что один уклон не описывает форму поверхности полностью.",
         "Оставлены полные окна внутри блоков с ≥95% валидных ячеек. Прибрежные, узкие и некоторые полярные "
         "области представлены хуже; эта популяция отличается от всех ячеек графика 3. "
         f"Суммарная площадь окон — примерно {stats['Earth']['patch_area_million_km2']/stats['Earth']['domain_area_million_km2']*100:.1f}% "
         f"земного домена и {stats['Mars']['patch_area_million_km2']/stats['Mars']['domain_area_million_km2']*100:.1f}% марсианского. "
         "Масштаб неровности здесь 100 км, а в предыдущем пилоте был 10 км.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    correlations = {}
    for ax, body in zip(axes, stats):
        p = patches[body]
        corr = weighted_spearman(p[FEATURES].to_numpy(), p.area_km2.to_numpy())
        correlations[body] = corr
        pd.DataFrame(corr, index=FEATURES, columns=FEATURES).to_csv(OUT / f"{body.lower()}_weighted_spearman.csv")
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(body)
        ax.set_xticks(range(5), SHORT, rotation=35, ha="right")
        ax.set_yticks(range(5), SHORT)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes.tolist(), shrink=.7, label="Area-weighted Spearman (weighted mid-CDF ranks)")
    save(fig, "05_global_correlations.png", "5. Корреляции: какие признаки дублируют друг друга",
         "Значения близко к +1 означают, что два признака растут вместе; около 0 — отсутствие "
         "выраженной монотонной связи. Корреляции рассчитаны отдельно по планетам с весами площади.",
         f"Связь медианного уклона с перепадом высот: ρ={correlations['Earth'][0,3]:.2f} для Земли "
         f"и ρ={correlations['Mars'][0,3]:.2f} для Марса. "
         "При подборе весов нужно учитывать зависимость признаков: иначе одно свойство рельефа может учитываться несколько раз.",
         "Корреляция не означает причинность и не доказывает, что признак бесполезен. "
         "Соседние окна пространственно зависимы; статистическая значимость и точность модели этим графиком не оцениваются.")

    xcap = max(float(np.log1p(patches[b].median_slope_deg.max())) for b in stats)
    ycap = max(float(np.log1p(patches[b].plane_residual_sd_m.max())) for b in stats)
    xbins, ybins = np.linspace(0, xcap, 45), np.linspace(0, ycap, 45)
    densities = {}
    for body in stats:
        p = patches[body]
        densities[body], _, _ = np.histogram2d(np.log1p(p.median_slope_deg), np.log1p(p.plane_residual_sd_m),
                                             bins=[xbins, ybins], weights=p.area_km2/p.area_km2.sum())
    overlap = float(np.minimum(densities['Earth'], densities['Mars']).sum())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, body in zip(axes, stats):
        density = np.ma.masked_where(densities[body].T <= 0, densities[body].T)
        im = ax.imshow(np.ma.log10(density), origin="lower", extent=(0, xcap, 0, ycap),
                       aspect="auto", cmap="viridis", vmin=-5, vmax=-1)
        xt, yt = np.linspace(0, xcap, 5), np.linspace(0, ycap, 5)
        ax.set_xticks(xt, [f"{v:.2g}" for v in np.expm1(xt)])
        ax.set_yticks(yt, [f"{v:.2g}" for v in np.expm1(yt)])
        ax.set_xlabel("Median slope (deg; log1p spacing)")
        ax.set_ylabel("Plane residual SD (m; log1p spacing)")
        ax.set_title(body)
    fig.colorbar(im, ax=axes.tolist(), shrink=.8, label="log10(share of retained patch area in bin)")
    save(fig, "06_global_feature_space.png", "6. Пространство признаков: есть ли общие типы рельефа",
         "Два признака рассматриваются одновременно. Чем ярче ячейка, тем большая доля площади "
         "окон имеет такое сочетание уклона и неровности. Координаты и цветовая шкала одинаковы.",
         f"Пересечение двух нормированных гистограмм на этой сетке составляет {overlap:.3f} "
         "(сумма минимумов масс ячеек). Общие области пространства признаков дают основание "
         "искать кандидатов по рельефу; редкие сочетания требуют отдельного рассмотрения.",
         "Величина зависит от разбиения на интервалы и выбранных двух признаков. "
         "Это не доля пригодных аналогов, не качество ML и не вероятность успешного испытания. "
         "Совпадение здесь не учитывает грунт, геологию, атмосферу и гравитацию.")

    metadata = json.loads((DATA / "manifest.json").read_text())
    total_bytes = sum(r["bytes"] for r in metadata)
    summary = pd.DataFrame(stats).T
    intro = (
        "Глобальный EDA рельефа Марса и земной суши. Полностью скачаны исходные глобальные файлы: "
        "ETOPO 2022 v1, 60 угловых секунд, ice surface, и MOLA MEGDR, 32 пикселя/градус. "
        f"Объём источников и масок — {total_bytes/1e6:.1f} МБ. В двух DEM 299 635 200 исходных ячеек. "
        "Проанализированы все 648 географических блоков каждой планеты. "
        "Это глобальное покрытие на более грубом масштабе, а не анализ каждого исходного пикселя как независимого наблюдения."
    )
    method = (
        "В каждом блоке 10° × 10° построена локальная AEQD-проекция и сетка с шагом 10 км. "
        "Высоты усредняются при перепроецировании; уклон считается по соседним высотам на этой сетке. "
        "Для Земли используется WGS84, для Марса сфера 3 396 000 м из PDS4-метаданных MEG032. "
        "Площади корректируются на areal_scale проекции; различия масштаба в пределах блока малы, но не равны нулю. "
        "Границы географических блоков назначаются по центрам ячеек: это глобальная дискретизация, "
        "а не бесшовная точная квадратура поверхности. Сумма площадей центровых ячеек для Марса "
        "примерно на 0.6% выше аналитической площади его сферы из-за дискретизации границ; "
        "приведённые площади приблизительны. Высокие широты не получают лишнего веса. "
        "Океаны и озёра Земли исключены по Natural Earth 1:10m ДО усреднения высот. Суша ниже уровня моря сохраняется. "
        "Побережья дополнительно отсекаются для уклонов. На Земле описана поверхность льда, а не подлёдный рельеф. "
        "Абсолютные высоты планет имеют разные нулевые уровни и в признаки сходства не входят."
    )
    patch_method = (
        "В каждой локальной сетке перечислены все неперекрывающиеся окна 100 × 100 км по выбранной привязке. "
        "Оставлены окна, полностью лежащие в географическом блоке и имеющие не менее 95% валидных ячеек. "
        "В каждом: медиана/90-й перцентиль уклона, доля уклонов >5°, p95−p05 высот, SD остатков от плоскости. "
        "Частичные окна у границ, побережий и узких полярных секторов исключены: "
        "глобальная статистика всех ячеек и статистика 100-км окон имеют разный охват. "
        "Процентили внутри окна невзвешенные (небольшая локальная вариация площади); "
        "распределения между окнами взвешены по площади. Порог 5° — параметр описания, а не предел ровера."
    )
    limits = (
        "На масштабе 10 км не видны камни, мелкие кратеры и уклоны под колесом. "
        "Данные не подтверждают безопасность посадки, проходимость или пригодность базы. "
        "MOLA содержит интерполяцию, ETOPO объединяет источники разной точности; QA исходных измерений здесь не оценивается. "
        "Глобальное покрытие устраняет выбор только знакомых районов, но не устраняет ограничения разрешения, масок и измерений. "
        "Следующий этап: глобальный поиск кандидатов по крупному рельефу, затем повторный анализ отобранных мест "
        "на более детальных DEM, проверка чувствительности к масштабу/границам и экспертная оценка задачи испытаний."
    )
    if (ROOT / "data/candidates/experiment.json").exists():
        limits = limits.split("Следующий этап:")[0] + (
            "Глобальный поиск кандидатов, пересчёт на подробных DEM и проверки масштаба/границ "
            "выполнены в продолжении ниже. Экспертная оценка задачи испытаний остаётся необходимой."
        )
    lines = ["# Глобальный EDA: Марс и земная суша", "", intro, "", "## Метод", "", method, "", patch_method, "",
             "## Числа и охват", "", "```", summary.round(4).to_string(), "```", "",
             "Площадь, покрытая полными 100-км окнами: "
             f"Земля {stats['Earth']['patch_area_million_km2']:.2f} млн км², "
             f"Марс {stats['Mars']['patch_area_million_km2']:.2f} млн км². "
             "Это отдельный знаменатель для графиков 4–6; полярные/прибрежные окна представлены неполно.", ""]
    for e in explanations:
        lines += ["## "+e["title"], "", f"![{e['title']}]({e['file']})", "",
                  "**Что показано.** "+e["what"], "", "**Вывод по расчётам.** "+e["finding"], "",
                  "**Ограничение.** "+e["limit"], ""]
    lines += ["## Чего эти данные не доказывают", "", limits, "", "## Источники", "",
              "- [NOAA ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model).",
              "- [NASA MOLA MEGDR: продукты и метаданные](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html).",
              "- [Natural Earth land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/).",
              "- Прямые URL, время скачивания, HTTP-метаданные и SHA-256: `data/global_terrain/manifest.json`.",
              "- Формулы и параметры: `global_terrain_extract.py`, `data/global_terrain/analysis_config.json`.", ""]
    (OUT / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")
    head = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>Глобальный EDA рельефа</title>"
            "<style>body{font:17px/1.65 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#f4f7fb;color:#142a3c}"
            "section{background:white;padding:28px;margin:28px 0;border-radius:14px}img{width:100%}"
            "table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd8e0}"
            "a{color:#176c97}strong{color:#143e57}</style><h1>Марс и Земля: глобальный EDA рельефа</h1>")
    sections = "<p>"+html.escape(intro)+"</p><section><h2>Метод и масштаб</h2><p>"+html.escape(method)+"</p><p>"+html.escape(patch_method)+"</p></section>"
    sections += "<section><h2>Объём и охват</h2>"+summary.round(4).to_html()+"</section>"
    for e in explanations:
        sections += "<section><h2>"+html.escape(e["title"])+"</h2>"
        sections += f"<img alt='{html.escape(e['title'])}' src='data:image/png;base64,{base64.b64encode((OUT/e['file']).read_bytes()).decode()}'>"
        for label, key in [("Что показано", "what"), ("Вывод по расчётам", "finding"), ("Ограничение", "limit")]:
            sections += f"<p><strong>{label}.</strong> {html.escape(e[key])}</p>"
        sections += "</section>"
    sections += "<section><h2>Как использовать для проекта</h2><p>"+html.escape(limits)+"</p>"
    sections += "<p>Источники: <a href='https://www.ncei.noaa.gov/products/etopo-global-relief-model'>NOAA ETOPO</a>, "
    sections += "<a href='https://pds-geosciences.wustl.edu/missions/mgs/megdr.html'>NASA MOLA</a>, "
    sections += "<a href='https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/'>Natural Earth</a>.</p></section>"
    from candidate_report import build_candidate_section
    candidate_html, candidate_markdown = build_candidate_section()
    if candidate_html:
        head += "<p><a href='#candidate-study'>Перейти к новым результатам: поиск и проверка кандидатов</a></p>"
        with (OUT / "REPORT_RU.md").open("a", encoding="utf-8") as file:
            file.write("\n\n"+candidate_markdown)
    from rover_report import build_rover_section
    rover_html, rover_markdown = build_rover_section()
    if rover_html:
        head += "<p><a href='#rover-study'>Новый этап: типы поверхности, недостающие данные и обучение ML</a></p>"
        with (OUT / "REPORT_RU.md").open("a", encoding="utf-8") as file:
            file.write("\n\n"+rover_markdown)
    from mars_types_report import build_mars_types_section
    mars_html, mars_markdown = build_mars_types_section()
    if mars_html:
        head += "<p><a href='#mars-types'>Демо: 10 типов Марса, клик по карте и фиксированные земные места</a></p>"
        with (OUT / "REPORT_RU.md").open("a", encoding="utf-8") as file:
            file.write("\n\n"+mars_markdown)
    from mars_match_report import build_match_section
    match_html, match_markdown = build_match_section()
    if match_html:
        head += "<p><a href='#match-method'>Для защиты: как модель выбрала земные места, паспорта совпадений и текст выступления</a></p>"
        with (OUT / "REPORT_RU.md").open("a", encoding="utf-8") as file:
            file.write("\n\n"+match_markdown)
    (OUT / "index.html").write_text(head+sections+candidate_html+rover_html+mars_html+match_html+"</html>", encoding="utf-8")
    (OUT / "figure_explanations.json").write_text(json.dumps(explanations, ensure_ascii=False, indent=2))
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2))
    print(summary.round(4).to_string())
    print(f"Report: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
