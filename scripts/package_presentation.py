"""Create a portable PowerPoint with speaker notes, then refresh checksums and ZIP.

Requires python-pptx==1.0.2. It is a presentation-build dependency only.
"""
import hashlib
import json
from pathlib import Path
import re
import zipfile

from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation'
notes = (OUT / 'speaker_notes_ru.md').read_text()
sections = re.split(r'\n## (\d{2})\. ', notes)
note_by_number = {int(sections[i]): sections[i+1].split('\n## Источники')[0] for i in range(1, len(sections), 2)}
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(16), Inches(9)
prs.core_properties.title = 'Earth analogues — данные, метод и объяснимость'
prs.core_properties.subject = 'Full-atlas EDA, terrain retrieval and exact SHAP'
prs.core_properties.author = 'NASA Space Apps project team'
for index, image in enumerate(sorted((OUT / 'figures').glob('0[1-8]_*.png')), 1):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(str(image), 0, 0, width=prs.slide_width, height=prs.slide_height)
    slide.notes_slide.notes_text_frame.text = note_by_number[index]
prs.save(OUT / 'project-slides-light.pptx')
check = Presentation(OUT / 'project-slides-light.pptx')
assert len(check.slides) == 8
assert all(slide.notes_slide.notes_text_frame.text for slide in check.slides)
files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix != '.zip' and p.name != 'manifest.json')
manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
with zipfile.ZipFile(OUT/'presentation-figures-light.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.zip': archive.write(p, p.relative_to(OUT))
print(json.dumps({'slides': len(check.slides), 'notes': len(note_by_number), 'powerpoint': str(OUT/'project-slides-light.pptx')}))
