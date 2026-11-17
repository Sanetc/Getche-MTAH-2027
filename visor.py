"""
More Than a Hack 2027 — visor del tracking con la posesión predicha
===================================================================

Genera un HTML autocontenido (sin internet) que anima el tracking: jugadores
de cada equipo, balón y la etiqueta de posesión fotograma a fotograma, con una
línea de tiempo de la predicción (y de la verdad, si se da) para saltar a
cualquier momento.

    python -I visor.py --data ..\\datos\\oficiales\\data --juego tactical \\
        --entrega entregas\\Getche.csv \\
        --verdad ..\\datos\\mtah-2027-oficial\\evaluation\\gt\\BAR-ATH_possession_gt_public.csv \\
        --out ..\\datos\\visor\\tactical.html

El HTML lleva dentro datos del partido: se genera fuera del repo (../datos) y
NO se sube a ningún sitio (ni git, ni la nube, ni un LLM).
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402

FALTA = -32768   # int16 para "no visible"


def rachas(etiquetas) -> list:
    """[[inicio, fin, etiqueta], ...] con fin exclusivo, en fotogramas del video."""
    e = np.asarray(etiquetas)
    c = np.flatnonzero(e[1:] != e[:-1]) + 1
    a, b = np.r_[0, c], np.r_[c, len(e)]
    return [[int(i), int(j), str(e[i])] for i, j in zip(a, b)]


def leer_etiquetas(ruta: Path, n: int, columna="label") -> np.ndarray:
    t = pd.read_csv(ruta)
    out = np.full(n, "X", dtype=object)
    f = t["frame"].to_numpy().astype(int)
    m = (f >= 0) & (f < n)
    out[f[m]] = t[columna].to_numpy()[m]
    return out


def a_int16(x: np.ndarray) -> np.ndarray:
    """Metros → decímetros en int16, NaN → FALTA."""
    y = np.where(np.isnan(x), FALTA, np.round(x * 10))
    return y.astype(np.int16)


def construir(df, pred, verdad, fases, cada, desde, hasta, titulo) -> str:
    fr = df["Frame"].to_numpy().astype(int)
    sel = (fr >= desde) & (fr <= hasta) & ((fr - desde) % cada == 0)
    d = df.loc[sel]
    cols_h, cols_a = D.columnas_jugadores(d, "H_"), D.columnas_jugadores(d, "A_")
    xs = cols_h + cols_a + ["ball_x"]
    X = d[xs].to_numpy(dtype=float)
    Y = d[[c[:-2] + "_y" for c in xs]].to_numpy(dtype=float)
    pos = np.stack([a_int16(X), a_int16(Y)], axis=2)          # (T, S, 2)
    datos = dict(
        titulo=titulo, fps=D.FPS, cada=cada, largo=D.LARGO, ancho=D.ANCHO, falta=FALTA,
        frames=d["Frame"].astype(int).tolist(), n_h=len(cols_h), n_a=len(cols_a),
        pos=base64.b64encode(pos.tobytes()).decode(),
        pred=rachas(pred[desde:hasta + 1]), desde=desde, hasta=hasta,
        verdad=None if verdad is None else [r for r in rachas(verdad[desde:hasta + 1]) if r[2] != "X"],
        fases=fases,
    )
    for clave in ("pred", "verdad"):
        if datos[clave]:
            for r in datos[clave]:
                r[0] += desde
                r[1] += desde
    return PLANTILLA.replace("__DATOS__", json.dumps(datos, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="carpeta con las subcarpetas de los partidos")
    ap.add_argument("--juego", required=True)
    ap.add_argument("--entrega", type=Path, required=True, help="CSV frame,label (la predicción)")
    ap.add_argument("--columna", default="label", help="columna de la predicción en --entrega")
    ap.add_argument("--verdad", type=Path, help="CSV frame,label con la verdad (opcional)")
    ap.add_argument("--columna-verdad", default="label")
    ap.add_argument("--desde", type=int, help="primer fotograma (por defecto, el primero con tracking)")
    ap.add_argument("--hasta", type=int, help="último fotograma")
    ap.add_argument("--cada", type=int, default=2, help="un fotograma de cada N (2 → 12,5 fps)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    carpeta = args.data / args.juego
    df = D.cargar_tracking(carpeta, args.juego, normalizar=False)
    n = int(df["Frame"].max()) + 1
    pred = leer_etiquetas(args.entrega, n, args.columna)
    verdad = leer_etiquetas(args.verdad, n, args.columna_verdad) if args.verdad else None
    fases = []
    if D.tiene_smart_tagging(carpeta, args.juego):
        st = D.cargar_smart_tagging(carpeta, args.juego)
        nombre_eq = {D.LOCAL: "H", D.VISITA: "A"}
        for f in st.itertuples():
            texto = f.codigo + (f" · {f.etiquetas.get('Type', '')}" if f.codigo == "SET PIECES" else "")
            fases.append([int(f.inicio), int(f.fin), texto, nombre_eq.get(f.equipo, "")])
    desde = args.desde if args.desde is not None else int(df["Frame"].min())
    hasta = args.hasta if args.hasta is not None else int(df["Frame"].max())
    html = construir(df, pred, verdad, fases, args.cada, desde, hasta, f"{args.juego} — {args.entrega.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Visor: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, fotogramas {desde}–{hasta} cada {args.cada})")


PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visor de posesión</title>
<style>
:root{--fondo:#111418;--panel:#1b2027;--texto:#e6e9ee;--suave:#8a94a3;--cesped:#1f5130;--linea:#d9e6dc;
 --H:#3b82f6;--A:#ef4444;--D:#9ca3af;--X:#2a2f37;--balon:#facc15}
*{box-sizing:border-box}body{margin:0;background:var(--fondo);color:var(--texto);font:14px system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:12px 16px}h1{font-size:16px;margin:0 0 8px;color:var(--suave);font-weight:500}
canvas{display:block;border-radius:6px;margin:0 auto}#linea{width:100%}
#controles{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
button,select,input[type=number]{background:var(--panel);color:var(--texto);border:1px solid #2f3742;border-radius:4px;padding:5px 9px;font:inherit}
button{cursor:pointer}#info{font-variant-numeric:tabular-nums}
.etq{display:inline-block;min-width:3.2em;text-align:center;border-radius:4px;padding:2px 6px;font-weight:600}
#leyenda{color:var(--suave);font-size:12px;margin-top:6px}#fase{color:var(--suave)}
</style></head><body><main>
<h1 id="titulo"></h1>
<canvas id="campo"></canvas>
<div id="controles">
 <button id="play">▶</button>
 <select id="vel"><option value="0.5">×0,5</option><option value="1" selected>×1</option><option value="2">×2</option><option value="4">×4</option><option value="8">×8</option></select>
 <button id="atras">−5 s</button><button id="adelante">+5 s</button>
 <button id="cambio_ant">⟨ cambio</button><button id="cambio_sig">cambio ⟩</button>
 <label>ir a fotograma <input id="ir" type="number" style="width:7em"></label>
 <span id="info"></span>
</div>
<canvas id="linea"></canvas>
<div id="leyenda">Línea de tiempo: arriba la predicción, abajo la verdad (si hay). <b style="color:var(--H)">H</b> local ·
 <b style="color:var(--A)">A</b> visitante · <b style="color:var(--D)">D</b> balón muerto. Clic para saltar; rueda del ratón para acercar.
 Teclas: espacio ▶/⏸, ← →. <span id="fase"></span></div>
</main>
<script>
const S=__DATOS__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL={H:css('--H'),A:css('--A'),D:css('--D'),X:css('--X')};
const raw=Uint8Array.from(atob(S.pos),c=>c.charCodeAt(0));const P=new Int16Array(raw.buffer);
const T=S.frames.length,NS=S.n_h+S.n_a+1;
document.getElementById('titulo').textContent=S.titulo;
const cv=document.getElementById('campo'),cx=cv.getContext('2d');
const lv=document.getElementById('linea'),lx=lv.getContext('2d');
let i=0,jugando=false,ultimo=0,acum=0,zoom=[S.desde,S.hasta+1];
function etiqueta(r,f){if(!r)return null;let a=0,b=r.length-1;while(a<=b){const m=(a+b)>>1;if(f<r[m][0])b=m-1;else if(f>=r[m][1])a=m+1;else return r[m][2];}return null;}
function fase(f){return S.fases.filter(x=>f>=x[0]&&f<x[1]).map(x=>x[2]+(x[3]?' ('+x[3]+')':'')).join(' | ');}
function tam(){const w=Math.min(cv.parentElement.clientWidth,innerHeight*0.66*S.largo/S.ancho);cv.style.width=w+'px';cv.width=w*devicePixelRatio;cv.height=w*S.ancho/S.largo*devicePixelRatio;
 lv.width=lv.clientWidth*devicePixelRatio;lv.height=(S.verdad?54:30)*devicePixelRatio;lv.style.height=(S.verdad?54:30)+'px';dibujar();}
function campo(){const W=cv.width,H=cv.height,k=W/S.largo;cx.fillStyle=css('--cesped');cx.fillRect(0,0,W,H);
 cx.strokeStyle=css('--linea');cx.lineWidth=1.5*devicePixelRatio;const m=(x,y)=>[(x+S.largo/2)*k,(S.ancho/2-y)*k];
 const r=(x0,y0,x1,y1)=>{const[a,b]=m(x0,y1),[c,d]=m(x1,y0);cx.strokeRect(a,b,c-a,d-b)};
 r(-S.largo/2,-S.ancho/2,S.largo/2,S.ancho/2);cx.beginPath();cx.moveTo(W/2,0);cx.lineTo(W/2,H);cx.stroke();
 cx.beginPath();cx.arc(W/2,H/2,9.15*k,0,7);cx.stroke();
 for(const s of[-1,1]){r(s*S.largo/2-(s>0?16.5:0),-20.16,s*S.largo/2+(s<0?16.5:0),20.16);r(s*S.largo/2-(s>0?5.5:0),-9.16,s*S.largo/2+(s<0?5.5:0),9.16);}
 return m;}
function dibujar(){const m=campo(),k=cv.width/S.largo,f=S.frames[i];
 for(let s=0;s<NS;s++){const o=(i*NS+s)*2,x=P[o],y=P[o+1];if(x===S.falta)continue;const[a,b]=m(x/10,y/10);
  cx.beginPath();if(s===NS-1){cx.fillStyle=css('--balon');cx.arc(a,b,0.6*k,0,7);}else{cx.fillStyle=s<S.n_h?COL.H:COL.A;cx.arc(a,b,1.0*k,0,7);}cx.fill();}
 const ep=etiqueta(S.pred,f),ev=etiqueta(S.verdad,f);
 const seg=(f-S.desde)/S.fps,mm=Math.floor(seg/60),ss=Math.floor(seg%60);
 document.getElementById('info').innerHTML=`fotograma ${f} · ${mm}:${String(ss).padStart(2,'0')} · predicción <span class="etq" style="background:${COL[ep]||COL.X}">${ep||'–'}</span>`+
  (S.verdad?` verdad <span class="etq" style="background:${COL[ev]||COL.X}">${ev||'–'}</span>`:'');
 document.getElementById('fase').textContent=fase(f)?'Smart Tagging: '+fase(f):'';
 linea(f);}
function linea(f){const W=lv.width,h=lv.height,[z0,z1]=zoom,k=W/(z1-z0);lx.fillStyle=css('--fondo');lx.fillRect(0,0,W,h);
 const filas=S.verdad?[S.pred,S.verdad]:[S.pred],alto=h/filas.length;
 filas.forEach((r,j)=>{for(const[a,b,e]of r){if(b<z0||a>z1)continue;lx.fillStyle=COL[e]||COL.X;lx.fillRect((a-z0)*k,j*alto+2,Math.max(1,(b-a)*k),alto-4);}});
 lx.fillStyle='#fff';lx.fillRect((f-z0)*k-1,0,2*devicePixelRatio,h);}
function ir(f){let a=0,b=T-1;while(a<b){const m=(a+b)>>1;if(S.frames[m]<f)a=m+1;else b=m;}i=a;dibujar();}
function paso(t){if(jugando){acum+=(t-ultimo)/1000*S.fps/S.cada*+document.getElementById('vel').value;const n=Math.floor(acum);
 if(n>0){acum-=n;i=Math.min(T-1,i+n);if(i===T-1)jugando=false;dibujar();}}ultimo=t;requestAnimationFrame(paso);}
function cambio(dir){const f=S.frames[i];const r=S.pred;let mejor=null;
 for(const[a]of r){if(dir>0&&a>f){mejor=a;break;}if(dir<0&&a<f)mejor=a;}if(mejor!==null)ir(dir<0&&mejor===f?mejor-1:mejor);}
document.getElementById('play').onclick=()=>{jugando=!jugando;document.getElementById('play').textContent=jugando?'⏸':'▶';};
document.getElementById('atras').onclick=()=>ir(S.frames[i]-5*S.fps);
document.getElementById('adelante').onclick=()=>ir(S.frames[i]+5*S.fps);
document.getElementById('cambio_ant').onclick=()=>cambio(-1);
document.getElementById('cambio_sig').onclick=()=>cambio(1);
document.getElementById('ir').onchange=e=>ir(+e.target.value);
lv.onclick=e=>{const r=lv.getBoundingClientRect();ir(zoom[0]+(e.clientX-r.left)/r.width*(zoom[1]-zoom[0]));};
lv.onwheel=e=>{e.preventDefault();const r=lv.getBoundingClientRect(),c=zoom[0]+(e.clientX-r.left)/r.width*(zoom[1]-zoom[0]);
 const w=Math.min(S.hasta+1-S.desde,Math.max(250,(zoom[1]-zoom[0])*(e.deltaY>0?1.4:0.7)));
 let a=Math.max(S.desde,c-(c-zoom[0])/(zoom[1]-zoom[0])*w);zoom=[a,Math.min(S.hasta+1,a+w)];dibujar();};
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.code==='Space'){e.preventDefault();document.getElementById('play').click();}
 if(e.code==='ArrowRight')ir(S.frames[i]+S.fps);if(e.code==='ArrowLeft')ir(S.frames[i]-S.fps);});
addEventListener('resize',tam);tam();requestAnimationFrame(t=>{ultimo=t;paso(t);});
</script></body></html>
"""

if __name__ == "__main__":
    main()
