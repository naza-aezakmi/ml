# Графики для презентации: данные → метод → объяснение

- [PowerPoint, 8 слайдов](project-slides-light.pptx)
- [PDF, 8 слайдов](project-slides-light.pdf)
- [Все PNG, SVG, таблицы и заметки одним архивом](presentation-figures-light.zip)
- [Что говорить к каждому графику](speaker_notes_ru.md)

Светлый фон, 16:9. Основные PNG — 3200 × 1800; отдельная небольшая вставка SHAP — `figures/09_shap_inset.png`. Векторные версии SVG приложены рядом. PowerPoint использует готовые изображения слайдов для одинакового отображения; численные данные и векторы доступны для отдельного редактирования.

EDA рассчитан по **всем 3 183 884 окнам действующего production-атласа**: Земля 1 356 072 после маски воды, Луна 380 956, Марс 1 446 856. Использована та же функция `terrainStats`, что и в API. Это анализ производных окон, не всех сырых наблюдений миссий. Spearman и квантили вычислены по всем окнам, без случайной подвыборки.

Графики ранжирования и SHAP основаны на реальном ответе поиска для Jezero; данные ответа сохранены в `data/actual_search_examples.json`. SHAP-вклады проверены на точное восстановление score. Диагностика KMeans на слайде 08 относится к отдельному обученному прототипу и явно отделена от текущего поиска.

## Пересоздать

Из `ml/`, при соседнем checkout `api/`, Node 22 и Python 3.12 с numpy/pandas/scipy/matplotlib:

```bash
node scripts/presentation_extract.cjs ../api ../reports/presentation-cache
python3.12 scripts/build_presentation.py --api ../api --cache ../reports/presentation-cache

# Необязательный PowerPoint, в отдельном окружении с python-pptx==1.0.2:
python scripts/package_presentation.py
```

Бинарный кеш всех характеристик остаётся вне Git. В репозитории — графики, агрегированные CSV, примеры ответов и SHA-256 в `manifest.json`. Источники и формулы: [карточка метода](../docs/MODEL_CARD.md), [происхождение данных](../docs/DATA_SOURCES.md).
