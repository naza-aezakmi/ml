"""Readable scientific explanation and presentation material for the prototype matcher."""
import base64,html,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from global_terrain_data import ROOT

OUT=ROOT/'reports/global_terrain'

def build_match_section():
    m=json.loads((ROOT/'public/mars_types.json').read_text())
    if 'matching_method' not in m:return '', ''
    method=m['matching_method'];types=m['types']
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    fields=method['fields'];labels=['Median slope (deg)','Slope p90 (deg)','Relative relief (m)']
    for ax,field,label in zip(axes,fields,labels):
        x=[t['retrieval_prototype'][field] for t in types for e in t['earth_candidates']]
        y=[e['features'][field] for t in types for e in t['earth_candidates']]
        ax.scatter(x,y,c=np.repeat(np.arange(10),3),cmap='tab10',s=45,alpha=.8)
        maximum=max(max(x),max(y))*1.05
        ax.plot([0,maximum],[0,maximum],color='gray',linestyle='--',linewidth=1)
        ax.set_xlabel('Mars learned prototype');ax.set_ylabel('Earth candidate')
        ax.set_title(label);ax.grid(alpha=.2)
    fig.suptitle('30 retrieved matches | common physical descriptors | dashed line = identical values')
    plot=OUT/'18_mars_earth_match_evidence.png';fig.savefig(plot,dpi=150,bbox_inches='tight');plt.close(fig)
    n=method['candidate_locations']
    summary=(f'Земные места выбраны алгоритмом поиска ближайших соседей среди {n:,} допустимых земных '
             'опорных ячеек. Запрос — обученный центр одного из десяти типов Марса. Входы обеих планет '
             'рассчитаны одной процедурой на сопоставимом километровом масштабе. Каждое совпадение '
             'снабжено исходными числами, расстоянием и точными вкладами признаков в остаточное различие.')
    steps=[('01 · Единое описание','MOLA для Марса и ETOPO для Земли → общий шаг около 10 км → окрестность из 81 ячейки. Абсолютные высоты разных планет не сравниваем.'),
           ('02 · Признаки','Медианный уклон описывает обычную крутизну, p90 — более крутые части, p95−p05 высот — амплитуду относительного рельефа.'),
           ('03 · Обучение','MiniBatchKMeans обучается на 150 000 марсианских примеров с учётом площади при выборке. Он находит центры профилей без выдуманных меток «правильный аналог».'),
           ('04 · Поиск на Земле',f'Все {n:,} земных опорных мест переводятся в те же шкалы. Алгоритм ранжирует расстояния до каждого центра и выбирает три места, разнесённых примерно на 500 км.'),
           ('05 · Проверка контекста','WorldCover показывает покрытие. Среди уже выбранной тройки выше отображаются места с ≥50% bare/sparse. Климат показан отдельно и не участвует в расстоянии.')]
    math_text=('zⱼ(x) = [ln(1 + xⱼ) − medianₘₐᵣₛ,ⱼ] / IQRₘₐᵣₛ,ⱼ\n'
               'd²(Earth, type) = Σⱼ [zⱼ(Earth) − c(type)ⱼ]²\n'
               'Вклад признака j = [zⱼ(Earth) − c(type)ⱼ]²')
    details=('Нормировка обучена один раз на марсианской обучающей выборке и без переобучения применяется '
             'к Земле. Это сохраняет общую шкалу. Логарифм смягчает влияние больших перепадов, IQR приводит '
             'метры и градусы к сопоставимому масштабу. Центр c — среднее в преобразованном пространстве. '
             'Поэтому его обратное преобразование не обязано совпадать с медианами класса, показанными '
             'в описательной таблице типов. В паспортах ниже используется именно центр реального поиска. '
             'Все три координаты пространства имеют равный коэффициент; два признака уклонов коррелированы, '
             'поэтому метрика является обоснованным baseline, а не выученной оценкой инженерной полезности.')
    section="<section id='match-method'><style>.match-flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.match-box{background:#eaf2f8;padding:18px;border-radius:12px}.match-passport{background:#f4f7fb;padding:18px;margin:14px 0;border-radius:12px}.match-pitch{border-left:5px solid #367d9d;padding:12px 24px;background:#edf4f8}</style>"
    section+='<h2>Почему эти места Земли соответствуют типам Марса</h2><p>'+summary+'</p><div class="match-flow">'
    for title,text in steps:section+='<div class="match-box"><strong>'+title+'</strong><p>'+text+'</p></div>'
    section+='</div><h3>Объяснимая математическая модель</h3><pre style="white-space:pre-wrap">'+html.escape(math_text)+'</pre><p>'+details+'</p>'
    section+='<p><strong>Как читать вклад:</strong> большой вклад означает причину несходства. Например, 70% '
    section+='вклада уклона — это 70% остаточного квадрата расстояния, а не «уклон обеспечил 70% сходства». '
    section+='Расстояние не является вероятностью и не переводится в проценты пригодности. SHAP для этой точной суммы не нужен.</p>'
    section+=f"<img alt='Сравнение физических признаков прототипов и земных кандидатов' src='data:image/png;base64,{base64.b64encode(plot.read_bytes()).decode()}'>"
    section+='<p>На графике — все 30 отобранных соответствий. Близость к диагонали подтверждает сходство '
    section+='измеренных признаков. Это проверка корректности поиска, а не независимая проверка проходимости или грунта.</p>'
    section+='<p>У наиболее пересечённого типа 10 отклонения заметнее: земная выборка не полностью '
    section+='воспроизводит экстремальный марсианский профиль. В паспортах явно отмечены случаи, где '
    section+='относительное отличие хотя бы одного признака ≥20%. Это порог наглядности, а не инженерный допуск.</p>'
    section+='<h3>Паспорта соответствий: раскрывай любой тип</h3>'
    md=['# Почему выбраны эти земные места','',summary,'',math_text,'',details,'']
    for t in types:
        section+='<details><summary><strong>'+html.escape(t['label'])+'</strong> — 3 объяснимых соответствия</summary>'
        for position,e in enumerate(t['earth_candidates'],1):
            expl=e['match_explanation']
            section+='<div class="match-passport"><h4>'+e['id']+f" · {e['lat']:.3f}°, {e['lon']:.3f}°</h4>"
            section+=f"<p>Расстояние {e['distance']:.6f}; ранг по рельефу до разнесения и сортировки покрытия: {expl['terrain_rank_before_diversity_and_cover']:,} из {n:,}. Позиция в каталоге: {position}.</p>"
            data=[{'Признак':p['label'],'Прототип Марса':round(p['mars_prototype_value'],4),
                   'Земля':round(p['earth_value'],4),'Единицы':p['unit'],
                   'Разница в общей шкале':round(p['signed_standardized_difference'],6),
                   'Доля остаточного различия, %':round(p['mismatch_share']*100,1)} for p in expl['comparisons']]
            section+=pd.DataFrame(data).to_html(index=False)+'<p>'+html.escape(expl['text_ru'])+'</p>'
            cover=e.get('landcover')
            if cover:
                section+=f"<p>Отдельная проверка покрытия: bare/sparse {cover['bare_sparse_fraction']:.1%}; растительность {cover['vegetated_fraction']:.1%}; снег/лёд {cover['snow_ice_fraction']:.1%}. Это контекст 2021 года.</p>"
            else:section+='<p>Покрытие: данных недостаточно; неизвестное значение не трактуется как голый грунт.</p>'
            if e['outside_mars_type_radius95']:section+='<p><strong>Частичное соответствие:</strong> расстояние больше 95-го перцентиля внутренних расстояний данного марсианского типа.</p>'
            section+='</div>'
            md.extend([e['id'],pd.DataFrame(data).to_string(index=False),expl['text_ru'],''])
        section+='</details>'
    pitch=('Мы помогаем команде миссии найти на Земле ландшафты для предварительных полевых испытаний. '
           'Мы извлекаем сопоставимые признаки из открытых моделей рельефа двух планет, обучаем компактный '
           'словарь марсианских ландшафтов и автоматически подбираем земные соответствия к каждому профилю. '
           'Нажатие на глобус открывает паспорт: какие параметры близки, что отличается и какие условия '
           'ещё не проверены. Весь каталог заранее вычислен, поэтому демо работает быстро и без внешних '
           'API при клике. Наша текущая версия сравнивает региональный рельеф; материалы поверхности, '
           'климат и инженерная проходимость имеют отдельные источники и ограничения. Это воспроизводимый '
           'первый этап выбора кандидатов для последующего обследования специалистами.')
    section+='<h3>Текст выступления на 45–60 секунд</h3><div class="match-pitch">'+pitch+'</div>'
    qa=[('Почему ровно десять?', 'Это выбранный уровень детализации демо. По silhouette шесть групп лучше разделены; десять дают более подробный словарь. Мы не называем десять универсальным законом геологии.'),
        ('Это нейросеть или таблица?', 'Типы обучены MiniBatchKMeans. Земные кандидаты рассчитаны ближайшими соседями. В интерфейсе хранится результат обучения и поиска; карта — способ доставки результата без тяжёлых вычислений.'),
        ('Почему не один и тот же участок для всего Марса?', 'Для каждого обученного центра есть свой каталог. Разные точки одного типа используют одинаковые представители; точная оптимальность для каждого клика не заявляется.'),
        ('Климат и почва совпадают?', 'Это не установлено. Подбор сделан по региональному рельефу. Температура и покрытие помогают увидеть ограничения; bare/sparse не является меткой песка или скалы.'),
        ('Как подтверждено качество?', 'Проверены исходные единицы, общий масштаб, точное воспроизведение расстояний, вкладов и работа кликов. Независимая полевая проверка аналогов и экспертная оценка остаются необходимыми.'),
        ('Чем этот ML отличается от обученного классификатора AI4Mars?', 'AI4Mars-модель распознаёт визуальные классы на снимках с марсохода. Она не определяет тип из координат и не использовалась для назначения грунта земным кандидатам.')]
    section+='<h3>Короткие ответы жюри</h3>'
    for question,answer in qa:section+='<details><summary>'+question+'</summary><p>'+answer+'</p></details>'
    section+='</section>'
    md+=['## Выступление','',pitch,'']+[q+'\n'+a for q,a in qa]
    (OUT/'PITCH_RU.md').write_text('\n\n'.join(md))
    return section,'\n\n'.join(md)
