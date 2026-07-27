import json, pathlib
S = pathlib.Path('/tmp/claude-1000/-disk2-xhand-inhand-xhand-inhand--claude-worktrees-wizardly-rhodes-b1dc15/70bf686a-b3a4-4e09-a8a4-0b5bb6f069e9/scratchpad')
pack = json.load(open(S/'mesh_pack.json'))
three = (S/'three.min.js').read_text()
orbit = (S/'OrbitControls.js').read_text()

html = """<title>锤子功能点标注</title>
<style>
  :root{
    --bg:#16181D; --panel:#1F232B; --line:#2C313B; --txt:#E8EAEE; --mut:#9AA3B2;
    --acc:#FF7A2F; --ok:#3ECF8E; --mono:ui-monospace,'SF Mono','JetBrains Mono',Consolas,monospace;
  }
  html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--txt);
    font:14px/1.5 system-ui,'PingFang SC','Microsoft YaHei',sans-serif}
  #view{position:fixed;inset:0;cursor:grab}
  #view.dragpt{cursor:crosshair}
  .chip{position:fixed;top:16px;left:16px;background:color-mix(in srgb,var(--panel) 88%,transparent);
    border:1px solid var(--line);border-radius:10px;padding:10px 14px;max-width:300px;backdrop-filter:blur(6px)}
  .chip h1{margin:0 0 4px;font-size:15px;font-weight:650;letter-spacing:.02em}
  .chip p{margin:0;color:var(--mut);font-size:12.5px}
  .chip b{color:var(--txt);font-weight:600}
  #panel{position:fixed;top:16px;right:16px;width:252px;background:color-mix(in srgb,var(--panel) 92%,transparent);
    border:1px solid var(--line);border-radius:12px;padding:14px 16px 16px;backdrop-filter:blur(6px);
    display:flex;flex-direction:column;gap:12px}
  .lab{font-size:11px;letter-spacing:.09em;color:var(--mut);text-transform:uppercase}
  table{border-collapse:collapse;width:100%;font-family:var(--mono);font-variant-numeric:tabular-nums}
  td{padding:3px 0;font-size:13.5px}
  td.ax{width:34px;color:var(--mut)}
  td.mm{text-align:right;color:var(--mut);font-size:12px}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:1px}
  .btnrow{display:flex;gap:8px}
  button{flex:1;border:1px solid var(--line);border-radius:8px;background:#262B34;color:var(--txt);
    padding:7px 0;font-size:13px;cursor:pointer;font-family:inherit}
  button:hover{border-color:#3A4150}
  button:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
  button.pri{background:var(--acc);border-color:var(--acc);color:#1A1005;font-weight:650}
  button.pri:hover{filter:brightness(1.07)}
  #toast{position:fixed;bottom:22px;left:50%;transform:translate(-50%,8px);background:var(--panel);
    border:1px solid var(--ok);color:var(--ok);border-radius:8px;padding:8px 16px;font-size:13px;
    opacity:0;transition:opacity .18s,transform .18s;pointer-events:none}
  #toast.on{opacity:1;transform:translate(-50%,0)}
  #json{position:absolute;left:-9999px}
  .views{display:flex;gap:6px}
  .views button{font-size:12px;padding:5px 0}
  @media (prefers-reduced-motion: reduce){ #toast{transition:none} }
</style>
<div id="view"></div>
<div class="chip">
  <h1>锤子功能点标注</h1>
  <p><b>拖拽橙色点</b>沿表面移动;<b>双击</b>网格任意处直接落点。<br>
     左键旋转 · 滚轮缩放 · 右键平移。坐标 = 仿真物体本体系(米)。</p>
</div>
<div id="panel">
  <div>
    <div class="lab" style="margin-bottom:6px"><span class="dot" style="background:var(--acc)"></span>食指功能点 · 本体系</div>
    <table>
      <tr><td class="ax">x</td><td id="px">–</td><td class="mm" id="pxmm"></td></tr>
      <tr><td class="ax">y</td><td id="py">–</td><td class="mm" id="pymm"></td></tr>
      <tr><td class="ax">z</td><td id="pz">–</td><td class="mm" id="pzmm"></td></tr>
    </table>
  </div>
  <div>
    <div class="lab" style="margin-bottom:6px">视角</div>
    <div class="views">
      <button id="v1">正</button><button id="v2">侧</button><button id="v3">顶</button><button id="v4">斜</button>
    </div>
  </div>
  <div class="btnrow">
    <button class="pri" id="copy">复制坐标 JSON</button>
    <button id="reset" style="flex:0 0 64px">重置</button>
  </div>
</div>
<div id="toast">已复制,粘贴给 Claude 即可保存</div>
<textarea id="json" readonly></textarea>
<script>__THREE__</script>
<script>__ORBIT__</script>
<script>
const PACK = __PACK__;
const DEFAULT_PT = PACK.default_point;
function decode(b64, T){ const s=atob(b64), a=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++) a[i]=s.charCodeAt(i); return new T(a.buffer); }
const q = decode(PACK.vb64, Uint16Array), fi = decode(PACK.fb64, Uint16Array);
const lo = PACK.lo, hi = PACK.hi, n = PACK.nv;
const verts = new Float32Array(n*3);
for(let i=0;i<n;i++) for(let k=0;k<3;k++)
  verts[i*3+k] = lo[k] + (q[i*3+k]/65535)*(hi[k]-lo[k]);
const idx = new Uint32Array(fi);

const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth, innerHeight);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x16181D);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.005, 10);
const ctr = new THREE.Vector3((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2);
camera.up.set(0,0,1);           // z-up: 与仿真一致
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(ctr);
function setView(dir){ camera.position.copy(ctr.clone().add(dir)); controls.update(); }
setView(new THREE.Vector3(0.28,-0.28,0.20));

scene.add(new THREE.HemisphereLight(0xdde3ee, 0x2a2d33, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 0.85); key.position.set(0.5,-0.3,0.8); scene.add(key);
const fill = new THREE.DirectionalLight(0x8899bb, 0.3); fill.position.set(-0.4,0.5,0.2); scene.add(fill);

const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(verts,3));
geo.setIndex(new THREE.BufferAttribute(idx,1));
geo.computeVertexNormals();
const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:0x8E939C, roughness:0.62, metalness:0.15}));
scene.add(mesh);

const grid = new THREE.GridHelper(0.6, 24, 0x3A4150, 0x242832);
grid.rotation.x = Math.PI/2; grid.position.set(ctr.x, ctr.y, lo[2]-0.004); scene.add(grid);
const axes = new THREE.AxesHelper(0.07); axes.position.set(lo[0]-0.02, lo[1]-0.02, lo[2]); scene.add(axes);

const marker = new THREE.Mesh(new THREE.SphereGeometry(0.004, 24, 16),
  new THREE.MeshStandardMaterial({color:0xFF7A2F, emissive:0xFF7A2F, emissiveIntensity:0.55, roughness:0.3}));
const halo = new THREE.Mesh(new THREE.SphereGeometry(0.0062, 24, 16),
  new THREE.MeshBasicMaterial({color:0xFF7A2F, transparent:true, opacity:0.22, depthWrite:false}));
marker.add(halo); scene.add(marker);
marker.position.fromArray(DEFAULT_PT);

const fmt = v => (v>=0?' ':'') + v.toFixed(4);
function refresh(){
  const p = marker.position;
  px.textContent=fmt(p.x); py.textContent=fmt(p.y); pz.textContent=fmt(p.z);
  pxmm.textContent=(p.x*1000).toFixed(1)+' mm'; pymm.textContent=(p.y*1000).toFixed(1)+' mm';
  pzmm.textContent=(p.z*1000).toFixed(1)+' mm';
  document.getElementById('json').value =
    JSON.stringify({index_point:[+p.x.toFixed(5),+p.y.toFixed(5),+p.z.toFixed(5)]});
}
refresh();

const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2(); let dragging=false;
function cast(ev){
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX-r.left)/r.width)*2-1, -((ev.clientY-r.top)/r.height)*2+1);
  ray.setFromCamera(ndc, camera);
}
renderer.domElement.addEventListener('pointerdown', ev=>{
  cast(ev);
  if(ray.intersectObject(marker,true).length){ dragging=true; controls.enabled=false; view.classList.add('dragpt'); }
});
addEventListener('pointermove', ev=>{
  if(!dragging) return; cast(ev);
  const hit = ray.intersectObject(mesh)[0];
  if(hit){ marker.position.copy(hit.point); refresh(); }
});
addEventListener('pointerup', ()=>{ if(dragging){ dragging=false; controls.enabled=true; view.classList.remove('dragpt'); }});
renderer.domElement.addEventListener('dblclick', ev=>{
  cast(ev); const hit = ray.intersectObject(mesh)[0];
  if(hit){ marker.position.copy(hit.point); refresh(); }
});

document.getElementById('copy').onclick = async ()=>{
  const t = document.getElementById('json').value;
  try{ await navigator.clipboard.writeText(t); }
  catch(e){ const ta=document.getElementById('json'); ta.style.left='0'; ta.select();
    document.execCommand('copy'); ta.style.left='-9999px'; }
  const toast=document.getElementById('toast'); toast.classList.add('on');
  setTimeout(()=>toast.classList.remove('on'), 1800);
};
document.getElementById('reset').onclick = ()=>{ marker.position.fromArray(DEFAULT_PT); refresh(); };
v1.onclick=()=>setView(new THREE.Vector3(0.36,0,0.06));
v2.onclick=()=>setView(new THREE.Vector3(0,-0.36,0.06));
v3.onclick=()=>setView(new THREE.Vector3(0.001,0,0.4));
v4.onclick=()=>setView(new THREE.Vector3(0.28,-0.28,0.20));

addEventListener('resize', ()=>{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight); });
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();
</script>
"""
html = html.replace('__PACK__', json.dumps(pack))
html = html.replace('__THREE__', three).replace('__ORBIT__', orbit)
out = S/'hammer_point_annotator.html'
out.write_text(html)
print('wrote', out, len(html)//1024, 'KB')
