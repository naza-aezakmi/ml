"""Add the practical click-map and taxonomy to the existing report."""
import base64,json
from pathlib import Path
import pandas as pd
from global_terrain_data import ROOT

def build_mars_types_section():
    path=ROOT/'public/mars_types.json'
    if not path.exists():return '', ''
    m=json.loads(path.read_text())
    binary=(ROOT/'public/mars_types_grid.bin').read_bytes()
    rows=[{'Тип':t['label'],'Доля площади, %':round(t['area_fraction']*100,2),
           'Медианный уклон, °':round(t['features_median']['median_slope_deg'],2),
           'p90 уклона, °':round(t['features_median']['p90_slope_deg'],2),
           'Перепад p95−p05, м':round(t['features_median']['relief_p95_p05_m'])} for t in m['types']]
    table=pd.DataFrame(rows).to_html(index=False)
    text=('Для демо весь Марс представлен 10 обученными прототипами рельефа. Типы получены MiniBatchKMeans '
          'по медиане уклона, p90 уклона и относительному перепаду высот. Дескрипторы вычислены для всех '
          'глобальных ячеек: 81 соседняя ячейка с шагом около 10 км, обычно окрестность радиусом около 50 км. '
          'Окрестности с самым дальним соседом дальше 65 км исключены. Идентификаторы типов упорядочены по '
          'выраженности рельефа; названия присвоены после обучения и не служили метками. '
          'Проверены k=6/10/14: silhouette примерно 0.410/0.356/0.312. Десять — выбор детализации для интерфейса, '
          'а не доказательство существования ровно десяти геологических классов.')
    workflow=('Клик по 3D-глобусу → перевод точки в широту/долготу → индекс готовой сетки 0.5° → typeId → '
              'фиксированный каталог трёх земных мест. Запрос не запускает ML, NASA API или загрузку DEM. '
              'Все точки одного типа возвращают один каталог. Температура и другие климатические значения '
              'подтягиваются из отдельного локального слоя по координатам, поэтому не одинаковы для всего типа.')
    limits=('Карта — глобальное приближение региональных форм рельефа; границы типов не являются физическими '
            'геологическими границами. При клике показаны типовые признаки кластера, а не точное измерение '
            'уклона в точке. Температура, давление и ветер — расчёт MCD 6.2 для Ls=0°, местного солнечного '
            'времени 12h, сценария climatology average solar, высоты 10 м для атмосферных переменных. '
            'Это не текущая погода и не годовая средняя. Влажность и осадки Марса не получены: null означает '
            '«нет данных», не ноль. Земные значения POWER — многолетняя климатология на других высотах; '
            'они не включены в расстояние и не объявляются климатическими аналогами. '
            'Скалистость, наличие песка, лёд, механика грунта и проходимость не выводятся из DEM. '
            'Покрытие новых земных мест показано отдельно: одинаковая форма рельефа может встречаться под лесом '
            'или льдом, поэтому список кандидатов не является рекомендацией готового полигона.')
    size=len(binary)+path.stat().st_size+(ROOT/'public/mars_analogs.mjs').stat().st_size
    section="<section id='mars-types'><h2>Демо к завтрашнему дню: нажми на Марс → получи тип и земные места</h2>"
    section+='<p>'+text+'</p><p>'+workflow+'</p>'
    section+=f'<p>Пакет для сайта: примерно {size/1024:.0f} КиБ без сжатия. Карта типов занимает {len(binary):,} байт. '
    section+='Скачать/скопировать файлы из public/: mars_types.json, mars_types_grid.bin, mars_analogs.mjs. '
    section+='<a href="../../public/mars_demo_bundle.zip">ZIP-пакет для команды</a>.</p>'
    section+="<canvas id='mars-type-map' width='720' height='360' style='width:100%;cursor:crosshair;border:1px solid #9aa'></canvas>"
    section+="<p>Широта <input id='mars-lat' type='number' min='-90' max='90' value='18.44' step='.01'> Долгота <input id='mars-lon' type='number' value='77.45' step='.01'> <button id='mars-go'>Показать</button></p><div id='mars-card'></div>"
    section+='<h3>Получившиеся десять типов</h3>'+table+'<p>'+limits+'</p>'
    code=(ROOT/'public/mars_analogs.mjs').read_text().replace('export ','')
    js="""
const catalog=META;
const labels=Uint8Array.from(atob('BINARY'),c=>c.charCodeAt(0));
const canvas=document.getElementById('mars-type-map'),ctx=canvas.getContext('2d');
const palette=['#234c7a','#357c9b','#66a9aa','#91bb93','#c6c67b','#dcb35f','#d99351','#c36b46','#974943','#612c43'];
const bitmap=ctx.createImageData(720,360);
for(let i=0;i<labels.length;i++){let h=palette[labels[i]];for(let j=0;j<3;j++)bitmap.data[4*i+j]=parseInt(h.slice(1+2*j,3+2*j),16);bitmap.data[4*i+3]=255;}
function line(parent,text){const p=document.createElement('p');p.textContent=text;parent.appendChild(p);}
function select(lat,lon){
 const card=document.getElementById('mars-card');card.replaceChildren();
 try{const r=lookupMars(lat,lon,catalog,labels);ctx.putImageData(bitmap,0,0);ctx.beginPath();ctx.arc((((lon+180)%360+360)%360)/.5,(90-lat)/.5,5,0,2*Math.PI);ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();
 line(card,`Марс ${lat.toFixed(2)}°, ${lon.toFixed(2)}° — ${r.type.label}`);
 const f=r.type.features_median;line(card,`Типовой профиль: медианный уклон ${f.median_slope_deg.toFixed(2)}°, p90 ${f.p90_slope_deg.toFixed(2)}°, перепад ${f.relief_p95_p05_m.toFixed(0)} м.`);
 if(r.climate){let c=r.climate;line(card,`MCD, Ls=0°, 12h: поверхность ${c.surface_temperature_c} °C; воздух на 10 м ${c.air_temperature_10m_c} °C; давление ${c.pressure_10m_pa} Па; ветер ${c.wind_10m_m_s} м/с. Модельный срез, не годовая средняя.`);}
 line(card,'Влажность/осадки Марса: нет полученных данных. Грунт и скалистость не определены.');
 for(const e of r.earthCandidates){line(card,`Земля ${e.id}: ${e.lat.toFixed(3)}°, ${e.lon.toFixed(3)}°; расстояние по рельефу ${e.distance.toFixed(3)}${e.outside_mars_type_radius95?' — за пределами типичного диапазона кластера':''}.`);
 if(e.match_explanation){line(card,e.match_explanation.text_ru);for(const p of e.match_explanation.comparisons)line(card,`${p.label}: прототип Марса ${p.mars_prototype_value.toFixed(2)} ${p.unit} → Земля ${p.earth_value.toFixed(2)} ${p.unit}; вклад в остаточное различие ${(p.mismatch_share*100).toFixed(1)}%.`);}
 if(e.landcover)line(card,`Покрытие 2021: голая/редкая поверхность ${(e.landcover.bare_sparse_fraction*100).toFixed(0)}%, растительность ${(e.landcover.vegetated_fraction*100).toFixed(0)}%, лёд ${(e.landcover.snow_ice_fraction*100).toFixed(0)}%.`);
 if(e.climate){let c=e.climate;line(card,`POWER, климатология Земли: воздух 2 м ${c.air_temperature_2m_c} °C; влажность ${c.relative_humidity_pct}%; осадки ${c.precipitation_mm_day} мм/день; ветер ${c.wind_10m_m_s} м/с. Климат не использован для выбора аналога.`);}}
 }catch(e){line(card,e.message);}}
canvas.addEventListener('click',e=>{let b=canvas.getBoundingClientRect();let lat=90-(e.clientY-b.top)/b.height*180,lon=(e.clientX-b.left)/b.width*360-180;document.getElementById('mars-lat').value=lat.toFixed(2);document.getElementById('mars-lon').value=lon.toFixed(2);select(lat,lon);});
document.getElementById('mars-go').onclick=()=>select(Number(document.getElementById('mars-lat').value),Number(document.getElementById('mars-lon').value));
select(18.44,77.45);
""".replace('META',json.dumps(m,ensure_ascii=False).replace('<','\\u003c')).replace('BINARY',base64.b64encode(binary).decode())
    section+='<script>'+code+'\n'+js+'</script></section>'
    md='# Глобальные типы Марса для клика\n\n'+text+'\n\n'+workflow+'\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\n'+limits
    return section,md
