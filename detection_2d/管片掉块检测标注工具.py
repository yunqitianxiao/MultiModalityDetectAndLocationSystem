#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管片掉块检测标注工具（YOLO 检测框 · 本地 Web 版）

针对 data/土建_管片掉块 中 人工标注.txt 所列的正样本图片，
在浏览器里手动框选"管片掉块(spalling)"目标，输出 YOLO 检测格式标注
（class cx cy w h，归一化坐标，单类 spalling=0）。

用法：
    python3 管片掉块检测标注工具.py            # 启动服务并自动打开浏览器
    python3 管片掉块检测标注工具.py --pack     # 直接把已有 labels 打包为 YOLO 数据集
    python3 管片掉块检测标注工具.py --port 9000

数据流：
    图片   data/土建_管片掉块/<name>.jpg
    清单   data/土建_管片掉块_人工标注.txt （每行一个图名，无扩展名）
    标注   data/土建_管片掉块/labels/<name>.txt   （YOLO 格式，空文件=无病害负样本）
    打包   data/土建_管片掉块_yolo_手工/{images,labels,data.yaml}

仅用 Python 标准库，无第三方依赖。
"""
import json
import os
import re
import shutil
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

# ---------------------------------------------------------------- 路径约定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "土建_管片掉块")
LIST_FILE = os.path.join(DATA_DIR, "土建_管片掉块_人工标注.txt")
LABEL_DIR = os.path.join(IMG_DIR, "labels")
PACK_DIR = os.path.join(DATA_DIR, "土建_管片掉块_yolo_手工")

CLASSES = ["spalling"]          # 单类：管片掉块
YOLO_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$"
)


# ---------------------------------------------------------------- 数据访问
def image_list():
    """返回待标注图片清单 [{name, url, done, annotated, boxCount}]，保持 txt 文件顺序。"""
    if not os.path.exists(LIST_FILE):
        return []
    result = []
    with open(LIST_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            base = raw.strip()
            if not base:
                continue
            name = base + ".jpg"
            if not os.path.exists(os.path.join(IMG_DIR, name)):
                continue
            label_path = os.path.join(LABEL_DIR, name.replace(".jpg", ".txt"))
            box_count = 0
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as lf:
                    text = lf.read()
                box_count = len([ln for ln in text.splitlines() if ln.strip()])
            result.append({
                "name": name,
                "url": "/api/image/" + name,
                "done": os.path.exists(label_path),
                "annotated": box_count > 0,
                "boxCount": box_count,
            })
    return result


def read_label(name):
    """读取某图的 YOLO 标注文本；无标注返回空串。"""
    safe = os.path.basename(name)
    p = os.path.join(LABEL_DIR, safe.replace(".jpg", ".txt"))
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def validate_yolo(text):
    """校验 YOLO 文本：每行 class + 4 个归一化坐标，class 必须合法，坐标在 [0,1]。
    空文本合法（负样本）。返回 (ok, 错误信息)。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        m = YOLO_LINE_RE.match(ln)
        if not m:
            return False, f"第{i + 1}行格式非法: {ln.strip()!r}"
        cls = int(m.group(1))
        if cls >= len(CLASSES):
            return False, f"第{i + 1}行 class={cls} 超出范围(0..{len(CLASSES) - 1})"
        for val in (float(m.group(2)), float(m.group(3)),
                    float(m.group(4)), float(m.group(5))):
            if not (0.0 <= val <= 1.0):
                return False, f"第{i + 1}行坐标越界: {ln.strip()!r}"
    return True, ""


def save_label(name, text):
    """保存标注；text 为空串则写空文件（负样本）。"""
    safe = os.path.basename(name)
    os.makedirs(LABEL_DIR, exist_ok=True)
    p = os.path.join(LABEL_DIR, safe.replace(".jpg", ".txt"))
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def delete_label(name):
    """删除标注文件，回退到未标状态。"""
    safe = os.path.basename(name)
    p = os.path.join(LABEL_DIR, safe.replace(".jpg", ".txt"))
    if os.path.exists(p):
        os.remove(p)


def pack_yolo():
    """把已标注图片(有 label 文件，含空负样本)打包为 YOLO 数据集目录。"""
    if os.path.exists(PACK_DIR):
        shutil.rmtree(PACK_DIR)
    img_out = os.path.join(PACK_DIR, "images")
    lbl_out = os.path.join(PACK_DIR, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    total = with_lb = without_lb = 0
    for item in image_list():
        name = item["name"]
        label_path = os.path.join(LABEL_DIR, name.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue                      # 未标注的跳过
        shutil.copy2(os.path.join(IMG_DIR, name), os.path.join(img_out, name))
        shutil.copy2(label_path, os.path.join(lbl_out, name.replace(".jpg", ".txt")))
        total += 1
        if item["boxCount"] > 0:
            with_lb += 1
        else:
            without_lb += 1

    yaml_text = (
        f"# 由 管片掉块检测标注工具.py 手工框标注导出(YOLO 检测框)\n"
        f"path: {PACK_DIR}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    with open(os.path.join(PACK_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)

    return {"total": total, "withLabels": with_lb, "withoutLabels": without_lb,
            "outDir": PACK_DIR}


# ---------------------------------------------------------------- HTTP 服务
PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>管片掉块检测标注工具</title>
<style>
  :root { --bd:#d9dee5; --bg:#f4f6f8; --acc:#2563eb; --ok:#16a34a; --warn:#d97706; --txt:#1f2937; }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; font:14px/1.5 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; color:var(--txt); background:var(--bg); }
  #app { display:flex; height:100%; }
  aside { width:270px; min-width:270px; background:#fff; border-right:1px solid var(--bd); display:flex; flex-direction:column; }
  header { padding:12px 14px; border-bottom:1px solid var(--bd); }
  header h1 { font-size:15px; margin:0 0 6px; }
  .progress { height:6px; background:#e5e7eb; border-radius:3px; overflow:hidden; }
  .progress > i { display:block; height:100%; width:0; background:var(--acc); transition:width .2s; }
  .meta { margin-top:6px; font-size:12px; color:#6b7280; }
  .toolbar { display:flex; flex-wrap:wrap; gap:6px; padding:10px 14px; border-bottom:1px solid var(--bd); }
  .toolbar button { padding:5px 10px; font-size:12px; border:1px solid var(--bd); border-radius:6px; background:#fff; cursor:pointer; }
  .toolbar button:hover { border-color:var(--acc); color:var(--acc); }
  .toolbar button.primary { background:var(--acc); border-color:var(--acc); color:#fff; }
  #list { flex:1; overflow-y:auto; }
  .item { display:flex; align-items:center; gap:8px; padding:7px 14px; cursor:pointer; border-bottom:1px solid #f0f1f3; }
  .item:hover { background:#f8fafc; }
  .item.active { background:#eff6ff; box-shadow:inset 3px 0 0 var(--acc); }
  .dot { width:8px; height:8px; border-radius:50%; background:#e5e7eb; flex:none; }
  .dot.ok { background:var(--ok); } .dot.neg { background:var(--warn); }
  .item .nm { flex:1; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .item .cnt { font-size:11px; color:#9ca3af; }
  main { flex:1; position:relative; }
  #cv { width:100%; height:100%; display:block; cursor:crosshair; }
  .hint { position:absolute; left:12px; bottom:12px; background:rgba(255,255,255,.92); border:1px solid var(--bd);
          border-radius:8px; padding:8px 12px; font-size:12px; color:#4b5563; box-shadow:0 1px 4px rgba(0,0,0,.08); }
  .hint b { color:var(--txt); }
  #status { position:absolute; right:12px; top:12px; background:rgba(255,255,255,.92); border:1px solid var(--bd);
            border-radius:8px; padding:6px 10px; font-size:12px; color:#374151; }
  #toast { position:fixed; left:50%; bottom:26px; transform:translateX(-50%) translateY(20px); background:#111827;
           color:#fff; padding:8px 16px; border-radius:8px; font-size:13px; opacity:0; transition:.25s; pointer-events:none; z-index:9; }
  #toast.show { opacity:.95; transform:translateX(-50%) translateY(0); }
</style>
</head>
<body>
<div id="app">
  <aside>
    <header>
      <h1>管片掉块检测标注</h1>
      <div class="progress"><i id="prog"></i></div>
      <div class="meta" id="meta">加载中…</div>
    </header>
    <div class="toolbar">
      <button id="btnNo" title="当前图无病害：写空标签并下一张">无病害</button>
      <button id="btnSkip" title="跳过：不保存，下一张">跳过</button>
      <button id="btnDel" title="删除当前图全部标注">删除标注</button>
      <button id="btnPack" class="primary" title="把已标注图片打包为 YOLO 数据集">导出YOLO</button>
    </div>
    <div id="list"></div>
  </aside>
  <main>
    <canvas id="cv"></canvas>
    <div id="status">—</div>
    <div class="hint"><b>左键拖拽</b>画框 · <b>中键/Space+拖拽</b>平移 · <b>滚轮</b>缩放<br>
      <b>Space/→</b>下一张 <b>←/Q</b>上一张 <b>Delete</b>删框 <b>S</b>保存 <b>Ctrl+Z</b>撤销 <b>Esc</b>取消</div>
  </main>
</div>
<div id="toast"></div>
<script>
"use strict";
const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');
const $ = id => document.getElementById(id);
const CLASSES = ['spalling'];
window.addEventListener('error', e=>{ console.error(e.error||e.message); });

let IMAGES = [], IDX = 0;
let img = null, imgW = 0, imgH = 0;
let boxes = [];            // {x,y,w,h} 图像像素坐标
let selIdx = -1;
let view = { s:1, ox:0, oy:0 };
let drag = null, pan = null, spaceDown = false;
let dirty = false, undoStack = [];
let saving = false;

/* ---------- 视图 ---------- */
function fitView(){
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!imgW || !imgH) return;
  view.s = Math.min(cw / imgW, ch / imgH) * 0.97;
  view.ox = (cw - imgW * view.s) / 2;
  view.oy = (ch - imgH * view.s) / 2;
}
function resizeCanvas(){
  canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
  if (img) { draw(); }
}
function draw(){
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!img) return;
  ctx.setTransform(view.s,0,0,view.s,view.ox,view.oy);
  ctx.drawImage(img, 0, 0);
  boxes.forEach((b,i)=>drawBox(b, i===selIdx, false));
  if (drag && drag.mode==='draw'){
    const x=Math.min(drag.x0,drag.x1), y=Math.min(drag.y0,drag.y1);
    const w=Math.abs(drag.x1-drag.x0), h=Math.abs(drag.y1-drag.y0);
    ctx.save(); ctx.setLineDash([6/view.s,5/view.s]); ctx.strokeStyle='#2563eb';
    ctx.lineWidth=1.5/view.s; ctx.strokeRect(x,y,w,h); ctx.restore();
  }
}
function drawBox(b, sel, deleting){
  ctx.save();
  ctx.strokeStyle = sel ? '#dc2626' : '#22c55e';
  ctx.fillStyle = sel ? 'rgba(220,38,38,.16)' : 'rgba(34,197,94,.12)';
  ctx.lineWidth = (sel?2.5:2)/view.s;
  ctx.fillRect(b.x,b.y,b.w,b.h);
  ctx.strokeRect(b.x,b.y,b.w,b.h);
  // 角点
  if (sel){
    const s=6/view.s; ctx.fillStyle='#dc2626';
    [[b.x,b.y],[b.x+b.w,b.y],[b.x,b.y+b.h],[b.x+b.w,b.y+b.h]].forEach(([px,py])=>{
      ctx.fillRect(px-s/2,py-s/2,s,s);
    });
  }
  // 类标签
  const fs=13/view.s; ctx.font=`${fs}px sans-serif`;
  const lbl=CLASSES[0]; const tw=ctx.measureText(lbl).width;
  const th=fs+4/view.s;
  ctx.fillStyle = sel ? '#dc2626' : '#16a34a';
  ctx.fillRect(b.x, Math.max(b.y-th,0), tw+10/view.s, th);
  ctx.fillStyle='#fff'; ctx.fillText(lbl, b.x+5/view.s, Math.max(b.y-th,0)+th-3/view.s);
  ctx.restore();
}

/* ---------- 坐标换算 ---------- */
function dev2img(mx,my){ return { x:(mx-view.ox)/view.s, y:(my-view.oy)/view.s }; }
function hitTest(ix,iy){
  const t=6/view.s;
  for (let i=boxes.length-1;i>=0;i--){
    const b=boxes[i];
    if (ix<b.x-t||ix>b.x+b.w+t||iy<b.y-t||iy>b.y+b.h+t) continue;
    const onL=Math.abs(ix-b.x)<=t, onR=Math.abs(ix-(b.x+b.w))<=t;
    const onT=Math.abs(iy-b.y)<=t, onB=Math.abs(iy-(b.y+b.h))<=t;
    let mode;
    if (onT&&onL) mode='nw'; else if (onT&&onR) mode='ne';
    else if (onB&&onL) mode='sw'; else if (onB&&onR) mode='se';
    else if (onT) mode='n'; else if (onB) mode='s';
    else if (onL) mode='w'; else if (onR) mode='e'; else mode='move';
    return { i, mode };
  }
  return null;
}
function clamp(v,a,b){ return Math.max(a, Math.min(b,v)); }

/* ---------- 交互 ---------- */
canvas.addEventListener('mousedown', e=>{
  if (!img) return;
  const p = dev2img(e.offsetX, e.offsetY);
  if (e.button===1 || (e.button===0 && spaceDown)){
    pan = { x0:e.offsetX, y0:e.offsetY, ox:view.ox, oy:view.oy };
    e.preventDefault(); return;
  }
  if (e.button!==0) return;
  pushUndo();
  const ht = hitTest(p.x, p.y);
  if (ht){ selIdx = ht.i; drag = { mode:ht.mode, ix:p.x, iy:p.y,
      bx:boxes[ht.i].x, by:boxes[ht.i].y, bw:boxes[ht.i].w, bh:boxes[ht.i].h }; }
  else { selIdx = -1; drag = { mode:'draw', x0:p.x, y0:p.y, x1:p.x, y1:p.y }; }
  draw();
});
canvas.addEventListener('mousemove', e=>{
  if (pan){
    view.ox = pan.ox + (e.offsetX-pan.x0);
    view.oy = pan.oy + (e.offsetY-pan.y0);
    draw(); return;
  }
  if (!drag) return;
  const p = dev2img(e.offsetX, e.offsetY);
  if (drag.mode==='draw'){
    drag.x1 = clamp(p.x,0,imgW); drag.y1 = clamp(p.y,0,imgH); dirty=true;
  } else {
    const b = boxes[drag.i]; if(!b){ drag=null; return; }
    const dx = p.x-drag.ix, dy = p.y-drag.iy;
    const L=drag.bx, T=drag.by, R=drag.bx+drag.bw, B=drag.by+drag.bh;
    let nl=L, nt=T, nr=R, nb=B;
    if (drag.mode==='move'){ nl=L+dx; nt=T+dy; nr=R+dx; nb=B+dy; }
    else if (drag.mode.includes('w')) nl=L+dx;
    else if (drag.mode.includes('e')) nr=R+dx;
    if (drag.mode.includes('n')) nt=T+dy;
    else if (drag.mode.includes('s')) nb=B+dy;
    if (drag.mode!=='move'){
      if (nr-nl<4){ if (drag.mode.includes('w')) nl=nr-4; else nr=nl+4; }
      if (nb-nt<4){ if (drag.mode.includes('n')) nt=nb-4; else nb=nt+4; }
      nl=clamp(nl,0,imgW); nr=clamp(nr,0,imgW); nt=clamp(nt,0,imgH); nb=clamp(nb,0,imgH);
    } else {
      const w=nr-nl, h=nb-nt;
      nl=clamp(nl,0,imgW-w); nt=clamp(nt,0,imgH-h);
      nr=nl+w; nb=nt+h;
    }
    b.x=nl; b.y=nt; b.w=nr-nl; b.h=nb-nt; dirty=true;
  }
  draw();
});
canvas.addEventListener('mouseup', ()=>{ drag=null; pan=null; draw(); });
canvas.addEventListener('mouseleave', ()=>{ drag=null; pan=null; });
canvas.addEventListener('wheel', e=>{
  e.preventDefault();
  if (!img) return;
  const f = e.deltaY<0 ? 1.12 : 1/1.12;
  const mx=e.offsetX, my=e.offsetY;
  const ix=(mx-view.ox)/view.s, iy=(my-view.oy)/view.s;
  view.s = clamp(view.s*f, 0.04, 60);
  view.ox = mx - ix*view.s; view.oy = my - iy*view.s;
  draw();
}, { passive:false });

/* ---------- 撤销 ---------- */
function pushUndo(){
  undoStack.push(JSON.stringify(boxes));
  if (undoStack.length>60) undoStack.shift();
}
function undo(){
  if (!undoStack.length) return;
  boxes = JSON.parse(undoStack.pop());
  selIdx=-1; dirty=true; draw();
}

/* ---------- 标注读写 ---------- */
function boxesToYolo(){
  return boxes.map(b=>{
    const cx=(b.x+b.w/2)/imgW, cy=(b.y+b.h/2)/imgH, w=b.w/imgW, h=b.h/imgH;
    return `0 ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
  }).join('\n');
}
function yoloToBoxes(text){
  const out=[];
  (text||'').split('\n').forEach(ln=>{
    const m=ln.trim().match(/^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)$/);
    if(!m) return;
    const cx=parseFloat(m[2])*imgW, cy=parseFloat(m[3])*imgH, w=parseFloat(m[4])*imgW, h=parseFloat(m[5])*imgH;
    out.push({x:cx-w/2, y:cy-h/2, w, h});
  });
  return out;
}
async function save(notify=true){
  if (saving) return;
  saving = true;
  try {
    const text = boxesToYolo();
    const r = await fetch('/api/label/'+encodeURIComponent(IMAGES[IDX].name),
                          { method:'POST', body:text });
    if (!r.ok){ const e=await r.json(); toast(e.error||'保存失败'); }
    else {
      dirty=false;
      IMAGES[IDX].done = true;
      IMAGES[IDX].annotated = text.length>0;
      IMAGES[IDX].boxCount = boxes.length;
      renderList(); renderStatus();
      if (notify) toast('已保存 ' + IMAGES[IDX].name);
    }
  } finally { saving=false; }
}

/* ---------- 图片切换 ---------- */
async function goto(i){
  if (i<0 || i>=IMAGES.length || i===IDX) return;
  if (dirty) await save(false);
  IDX=i; await loadImage();
}
async function loadImage(){
  img=null; draw(); renderStatus();
  const item = IMAGES[IDX];
  const lbl = await (await fetch('/api/label/'+encodeURIComponent(item.name))).text();
  const el = new Image();
  el.onload = ()=>{ img=el; imgW=el.naturalWidth; imgH=el.naturalHeight;
    boxes = yoloToBoxes(lbl); selIdx=-1; undoStack=[]; dirty=false;
    fitView(); draw(); renderList(); renderStatus(); };
  el.onerror = ()=> toast('图片加载失败 '+item.name);
  el.src = item.url;
}

/* ---------- UI ---------- */
function renderList(){
  const list=$('list'); list.innerHTML='';
  IMAGES.forEach((it, i)=>{
    const d=document.createElement('div');
    d.className='item'+(i===IDX?' active':'');
    const dot=document.createElement('span');
    dot.className='dot'+(it.annotated?' ok':(it.done?' neg':''));
    const nm=document.createElement('span'); nm.className='nm'; nm.textContent=it.name;
    const cnt=document.createElement('span'); cnt.className='cnt';
    cnt.textContent = it.done ? (it.annotated ? it.boxCount+'框' : '无病害') : '未标';
    d.appendChild(dot); d.appendChild(nm); d.appendChild(cnt);
    d.onclick = ()=> goto(i);
    list.appendChild(d);
  });
  const done = IMAGES.filter(it=>it.done).length;
  $('prog').style.width = (IMAGES.length? done/IMAGES.length*100:0)+'%';
  $('meta').textContent = `已标 ${done} / ${IMAGES.length}`;
}
function renderStatus(){
  const it = IMAGES[IDX];
  $('status').textContent = it ? `${IDX+1}/${IMAGES.length} · ${it.name} · ${boxes.length} 框` : '—';
}
let toastTimer;
function toast(msg){
  const t=$('toast'); t.textContent=msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>t.classList.remove('show'),1600);
}

/* ---------- 键位 ---------- */
document.addEventListener('keydown', e=>{
  if (e.key===' '){ e.preventDefault(); spaceDown=true; }
  if (e.key==='ArrowRight'||e.key===' '){ e.preventDefault(); goto(IDX+1); }
  else if (e.key==='ArrowLeft'||e.key==='q'||e.key==='Q'){ goto(IDX-1); }
  else if (e.key==='Delete'||e.key==='Backspace'){
    if (selIdx>=0 && selIdx<boxes.length){ pushUndo(); boxes.splice(selIdx,1); selIdx=-1; dirty=true; draw(); renderStatus(); }
  }
  else if (e.key==='Escape'){ drag=null; }
  else if (e.key==='s'||e.key==='S'){ e.preventDefault(); save(); }
  else if ((e.ctrlKey||e.metaKey) && (e.key==='z'||e.key==='Z')){ e.preventDefault(); undo(); }
});
document.addEventListener('keyup', e=>{ if(e.key===' ') spaceDown=false; });

/* ---------- 工具栏 ---------- */
$('btnNo').onclick = async ()=>{ pushUndo(); boxes=[]; await save(true); goto(IDX+1); };
$('btnSkip').onclick = ()=>{ dirty=false; goto(IDX+1); };
$('btnDel').onclick = async ()=>{
  const it=IMAGES[IDX];
  await fetch('/api/label/'+encodeURIComponent(it.name)+'/delete',{method:'POST'});
  boxes=[]; dirty=false; it.done=false; it.annotated=false; it.boxCount=0;
  renderList(); renderStatus(); draw(); toast('已删除标注');
};
$('btnPack').onclick = async ()=>{
  const r = await fetch('/api/pack',{method:'POST'});
  const j = await r.json();
  toast(`导出完成：${j.total} 张（含病害 ${j.withLabels}，负样本 ${j.withoutLabels}）→ ${j.outDir}`);
};

/* ---------- 启动 ---------- */
window.addEventListener('resize', resizeCanvas);
(async ()=>{
  IMAGES = await (await fetch('/api/images')).json();
  renderList();
  if (!IMAGES.length){ toast('未找到待标注图片'); return; }
  resizeCanvas();
  await loadImage();
})();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "SpallingLabeler/1.0"

    # -- 工具函数
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _image_path(self, name):
        """返回 IMG_DIR 下合法图片的绝对路径；非法返回 None（防路径穿越）。"""
        base = os.path.basename(name)
        p = os.path.abspath(os.path.join(IMG_DIR, base))
        if not p.startswith(os.path.abspath(IMG_DIR) + os.sep) and p != os.path.abspath(IMG_DIR):
            return None
        if not os.path.isfile(p):
            return None
        return p

    # -- 路由
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/" or path == "/index.html":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/images":
            self._json({"images": image_list()})
        elif path.startswith("/api/image/"):
            name = path[len("/api/image/"):]
            p = self._image_path(name)
            if p is None:
                self._json({"error": "image not found: " + name}, 404)
                return
            with open(p, "rb") as f:
                data = f.read()
            self._send(200, data, "image/jpeg")
        elif path.startswith("/api/label/"):
            name = path[len("/api/label/"):]
            self._send(200, read_label(name), "text/plain; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/pack":
            self._json(pack_yolo())
        elif path.startswith("/api/label/"):
            rest = path[len("/api/label/"):]
            if rest.endswith("/delete"):
                name = rest[:-len("/delete")]
                delete_label(name)
                self._json({"ok": True})
                return
            name = rest
            length = int(self.headers.get("Content-Length", 0) or 0)
            text = self.rfile.read(length).decode("utf-8") if length else ""
            ok, err = validate_yolo(text)
            if not ok:
                self._json({"error": err}, 400)
                return
            save_label(name, text)
            self._json({"ok": True, "saved": name})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        sys.stderr.write("[标注工具] " + (fmt % args) + "\n")


def free_port(start=8787, tries=30):
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


def main():
    if "--pack" in sys.argv:
        result = pack_yolo()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    port = None
    if "--port" in sys.argv:
        i = sys.argv.index("--port")
        port = int(sys.argv[i + 1])
    else:
        port = free_port()
    if port is None:
        print("找不到可用端口，请用 --port 指定。")
        sys.exit(1)

    if not os.path.exists(IMG_DIR):
        print("图片目录不存在: " + IMG_DIR)
        sys.exit(1)
    if not os.path.exists(LIST_FILE):
        print("清单文件不存在: " + LIST_FILE)
        sys.exit(1)
    n = len(image_list())
    print(f"待标注图片: {n} 张")
    if n == 0:
        print("清单为空或与图片无匹配，请检查清单内容。")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"标注工具已启动: {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
