"""Reproducible offline EDA of measured Earth/Mars terrain on a common 1 km grid."""
from __future__ import annotations

import base64
import html
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "analog-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from terrain_data import DATA, PIXEL_M, ROOT

OUT = ROOT / "reports/terrain"
FEATURES = ["median_slope_deg", "p90_slope_deg", "steep_fraction_gt5deg",
            "relief_p95_p05_m", "plane_residual_sd_m"]
LABELS = ["Median slope (deg)", "90th percentile slope (deg)", "Area with slope >5 deg (fraction)",
          "Relative relief p95-p05 (m)", "Plane residual SD (m; 10 km patches)"]
COLORS = {"Earth": "#24799e", "Mars": "#c25835"}


def slopes(z: np.ndarray, spacing: float = PIXEL_M) -> np.ndarray:
    dy, dx = np.gradient(z, spacing, spacing)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def features(z: np.ndarray, slope: np.ndarray, spacing: float = PIXEL_M) -> dict:
    valid, slope_valid = np.isfinite(z), np.isfinite(slope)
    if valid.mean() < .95 or slope_valid.mean() < .95:
        raise ValueError("Insufficient valid elevation/slope coverage; require >=95%")
    y, x = np.indices(z.shape, dtype=float)
    design = np.column_stack((x[valid]*spacing, y[valid]*spacing, np.ones(valid.sum())))
    coef = np.linalg.lstsq(design, z[valid], rcond=None)[0]
    residual = z[valid] - design @ coef
    return {"valid_fraction": float(valid.mean()), "valid_slope_fraction": float(slope_valid.mean()),
            "median_elevation_m": float(np.nanmedian(z)),  # descriptive only; datums differ
            "median_slope_deg": float(np.nanmedian(slope)),
            "p90_slope_deg": float(np.nanpercentile(slope, 90)),
            "steep_fraction_gt5deg": float((slope[slope_valid] > 5).mean()),
            "relief_p95_p05_m": float(np.nanpercentile(z, 95) - np.nanpercentile(z, 5)),
            "plane_residual_sd_m": float(np.std(residual))}


def main() -> None:
    manifest_path = DATA / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run terrain_data.py --download first")
    manifest = json.loads(manifest_path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows, regions, grids = [], [], []
    for record in manifest:
        with rasterio.open(ROOT / record["aligned_file"]) as src:
            assert src.res == (PIXEL_M, PIXEL_M), "Incorrect grid spacing"
            z = src.read(1, masked=True).filled(np.nan).astype(float)
        slope = slopes(z)
        center = z[3:-3, 3:-3]
        center_slope = slope[3:-3, 3:-3]
        grids.append((record, center, center_slope))
        base = {"site_id": record["id"], "body": record["body"],
                "center_lat": record["lat"], "center_lon": record["lon"],
                "pixel_m": PIXEL_M, "source_url": record["source_url"]}
        regions.append({**base, "window_km": 40, **features(center, center_slope)})
        for row in (0, 15, 30):
            for col in (0, 15, 30):
                ys, xs = slice(row, row+10), slice(col, col+10)
                rows.append({**base, "patch_id": f"{record['id']}_r{row}_c{col}",
                             "window_km": 10, "patch_center_east_m": -20000+(col+5)*1000,
                             "patch_center_north_m": 20000-(row+5)*1000,
                             **features(center[ys, xs], center_slope[ys, xs])})
    frame = pd.DataFrame(rows)
    regional = pd.DataFrame(regions)
    frame.to_csv(OUT / "patch_features.csv", index=False)
    regional.to_csv(OUT / "region_features.csv", index=False)
    frame.groupby("body")[FEATURES].agg(["min", "median", "max"]).to_csv(OUT / "body_summary.csv")
    pd.DataFrame(manifest).drop(columns=["source_crs_wkt", "analysis_crs_wkt"]).to_csv(OUT / "data_inventory.csv", index=False)
    plt.rcParams.update({"figure.facecolor": "#f4f7fb", "axes.facecolor": "white", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(OUT / name, bbox_inches="tight")
        plt.close(fig)

    for mode, filename in [("elevation", "01_relative_elevation_maps.png"), ("slope", "02_slope_maps.png")]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
        arrays = [z-np.nanmedian(z) if mode == "elevation" else s for _, z, s in grids]
        upper = float(np.nanpercentile(np.abs(np.concatenate([a.ravel() for a in arrays])), 99))
        for ax, (record, _, _), arr in zip(axes.flat, grids, arrays):
            im = ax.imshow(arr, extent=(-20, 20, -20, 20), cmap="RdBu_r" if mode == "elevation" else "magma",
                           vmin=-upper if mode == "elevation" else 0, vmax=upper)
            ax.set_title(f"{record['body']} / {record['id']}")
            ax.set_xlabel("East of center (km)")
            ax.set_ylabel("North of center (km)")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65,
                     label="Height relative to regional median (m)" if mode == "elevation" else "Slope (deg at 1 km grid)")
        fig.suptitle(f"{'Relative elevation' if mode == 'elevation' else 'Terrain slope'} | 40 x 40 km, 1 km grid\n"
                     "Common scale; color limits clip outer 1% for readability", fontsize=16)
        save(fig, filename)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, ax in enumerate(axes.flat):
        if j == 5:
            ax.axis("off")
            ax.text(0, .9, "54 Earth + 27 Mars patches\n10 x 10 km per patch\n9 patches per region, 5 km gaps\n"
                    "6 Earth / 3 Mars selected regions\nNearby patches remain dependent\nNot a global population sample", va="top")
            break
        ax.boxplot([frame.loc[frame.body == body, FEATURES[j]] for body in ("Earth", "Mars")],
                   tick_labels=["Earth", "Mars"], widths=.5)
        for i, body in enumerate(("Earth", "Mars"), 1):
            vals = frame.loc[frame.body == body, FEATURES[j]].to_numpy()
            ax.scatter(i+np.linspace(-.13, .13, len(vals)), vals, alpha=.5, s=15, color=COLORS[body])
        ax.set_title(LABELS[j])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Terrain features: selected patches on a common spatial grid", fontsize=16)
    save(fig, "03_feature_distributions.png")

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for record, _, s in grids:
        vals = np.sort(s[np.isfinite(s)])
        ax.plot(vals, np.arange(1, len(vals)+1)/len(vals), label=f"{record['body']}: {record['id']}",
                linestyle="--" if record["body"] == "Mars" else "-", linewidth=2)
    ax.set_xlabel("Slope (deg at 1 km grid)")
    ax.set_ylabel("Fraction of valid pixels at or below slope")
    ax.set_title("Regional slope distributions | 40 x 40 km per region")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.grid(alpha=.2)
    save(fig, "04_slope_ecdf.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    short = ["Slope median", "Slope p90", "Fraction >5 deg", "Relief", "Plane residual SD"]
    for ax, body in zip(axes, ("Earth", "Mars")):
        subset = frame.loc[frame.body == body, FEATURES]
        corr = subset.corr(method="spearman")
        corr.to_csv(OUT / f"spearman_{body.lower()}.csv")
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(5), short, rotation=40, ha="right")
        ax.set_yticks(range(5), short)
        ax.set_title(f"{body}: {len(subset)} spatially dependent patches")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes.tolist(), shrink=.7, label="Spearman correlation; exploratory")
    save(fig, "05_feature_correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for site_id, sub in frame.groupby("site_id", sort=False):
        body = sub.body.iloc[0]
        for ax, feature in zip(axes, ["relief_p95_p05_m", "plane_residual_sd_m"]):
            ax.scatter(sub.median_slope_deg, sub[feature], marker="^" if body == "Mars" else "o",
                       s=45, alpha=.8, label=f"{body}: {site_id}")
    for ax, ylabel in zip(axes, ["Relative relief p95-p05 (m)", "Plane residual SD (m)"]):
        ax.set_xlabel("Median slope (deg at 1 km grid)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.suptitle("Feature space | each point = 10 x 10 km patch; triangle = Mars")
    save(fig, "06_feature_space.png")

    lines = ["# EDA рельефа: Марс и Земля", "",
             "Численные DEM, без искусственных профилей и климатических прокси.", "",
             f"Обработано {len(regional)} районов: 3 на Марсе, 6 на Земле. "
             f"Извлечено {len(frame)} окон 10 × 10 км (9 на район).", "",
             "## Данные и масштаб", "",
             "- Марс: NASA MGS MOLA, USGS GeoTIFF, 128 пикселей/градус (~463 м у экватора).",
             "- Земля: Copernicus DEM GLO-90, DSM (~90 м), включает поверхность растительности и построек.",
             "- Каждый район: 40 × 40 км; локальная азимутальная равнопромежуточная проекция; пиксель 1 км.",
             "- Для Марса использована сфера радиуса 3 396 190 м, соответствующая исходному GeoTIFF; для Земли WGS84.",
             "- Перепроецирование с усреднением. Это согласует шаг и метод обработки, но не делает "
             "эффективное разрешение и погрешность двух источников одинаковыми.",
             "- Уклон: atan(sqrt((dz/dx)²+(dz/dy)²)), центральные разности numpy.gradient на сетке 1 км.",
             "- Уклоны вычислены до обрезки буфера 3 км. Окна 10 км не перекрываются, зазор 5 км.",
             "- Неровность: SD остатков после МНК-подгонки плоскости внутри окна. Для окон 10 и 40 км "
             "это характеристики разных масштабов; не смешивать соответствующие таблицы.",
             "- Порог 5° — исследовательский параметр, а не предел проходимости конкретного ровера.",
             "- Высота Марса относительно ареоида, Земли относительно EGM2008: абсолютные высоты "
             "сохранены для описания, в межпланетное сравнение признаков не входят.", "",
             "## Покрытие", "",
             f"Минимальная доля валидных высот в окне: {frame.valid_fraction.min():.1%}; "
             f"уклонов: {frame.valid_slope_fraction.min():.1%}.", "",
             "Валидность означает наличие конечного числа после маски NoData. Это не независимая "
             "проверка точности или QA исходных измерений. MOLA содержит интерполированные ячейки; "
             "карта количества измерений и вспомогательные QA-слои Copernicus здесь не загружались.", "",
             "## Что видно в выбранной выборке", ""]
    for body in ("Earth", "Mars"):
        sub = frame[frame.body == body]
        lines += [f"**{body}** ({len(sub)} окон):", ""]
        for feature, label in zip(FEATURES, LABELS):
            lines.append(f"- {label}: медиана {sub[feature].median():.3f}, "
                         f"диапазон {sub[feature].min():.3f} … {sub[feature].max():.3f}.")
        lines += [""]
    regional_sorted = regional.sort_values("median_slope_deg")
    lines += [f"Наименьший медианный уклон в районе {regional_sorted.iloc[0].site_id} "
              f"({regional_sorted.iloc[0].median_slope_deg:.2f}°), наибольший — "
              f"{regional_sorted.iloc[-1].site_id} ({regional_sorted.iloc[-1].median_slope_deg:.2f}°).", "",
              "Это наблюдения по выбранным координатам и масштабу, а не вывод о всей Земле или Марсе.", "",
              "## Графики", "",
              "1. Карты высот относительно медианы каждого района; общая цветовая шкала.",
              "2. Карты уклонов с общей шкалой.",
              "3. Распределения пяти признаков в окнах 10 км.",
              "4. ECDF уклонов по районам: доля площади ниже заданной крутизны.",
              "5. Корреляции отдельно для Земли и Марса: проверка дублирования признаков.",
              "6. Пространство признаков: уклон, перепад и неровность.", "",
              "## Ограничения и следующий эксперимент", "",
              "Районы выбраны вручную для разнообразия, земные центры обозначают приблизительные районы, "
              "а не обследованные полигоны. Марсианские окна тоже не являются официальными границами посадочных площадок. "
              "По 1 км пикселям нельзя судить о камнях, колёсной проходимости, грунте или безопасности посадки.", "",
              "81 окно не равно 81 независимому району. Соседние окна пространственно зависимы; "
              "в обучении и оценке необходимо разделение по районам. Формальные тесты различий планет "
              "и оценки глобального распределения на этой выборке не выполняются.", "",
              "Далее: сравнить формы распределений уклона и масштаб неровности; проверить чувствительность "
              "к сетке 0.75/1/2 км и смещению границ; выбрать веса с экспертом; расширить земные кандидаты "
              "и подключить QA. Только затем ранжировать земные аналоги конкретного запроса.", "",
              "Источники и прямые ссылки: [datasets_ru.md](datasets_ru.md). "
              "Полная геопривязка и происхождение каждого окна: `data/terrain/manifest.json`.", ""]
    (OUT / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")
    content = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>Марс и Земля — EDA рельефа</title>"
               "<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;background:#f4f7fb;color:#142a3c}"
               "img{width:100%;margin:20px 0}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;font-size:13px}"
               "td,th{padding:8px;border:1px solid #ccd8e0}</style><h1>Марс и Земля: EDA рельефа</h1>"
               "<p>9 районов · 81 окно · общий пиксель 1 км · численные DEM</p>"
               "<p><a href='../global_terrain/index.html'>Новый глобальный EDA Земли и Марса с объяснением каждого графика</a> · "
               "<a href='GRAPH_GUIDE_RU.md'>Как читать графики этого пилота</a></p>"
               + regional[["site_id", "body", *FEATURES]].round(3).to_html(index=False)
               + "<pre>" + html.escape("\n".join(lines)) + "</pre>")
    for path in sorted(OUT.glob("*.png")):
        content += f"<img alt='{html.escape(path.stem)}' src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'>"
    (OUT / "index.html").write_text(content+"</html>", encoding="utf-8")
    print(f"Terrain EDA complete: {len(frame)} patches; report: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
