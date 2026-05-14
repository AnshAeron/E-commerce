#!/usr/bin/env python3
"""Flask web UI for the E-Commerce CRS Assistant."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, Response, jsonify, request
from pipeline.crs import CRSPipeline

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = CRSPipeline()
    return _pipeline


app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\"/>
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>
<title>E-Commerce AI Assistant</title>
<style>
:root{--bg:#f6fbff;--surface:#ffffff;--card:#eef6ff;--accent:#2f7de1;--accent2:#ffad59;--sun:#ffe4bf;--text:#163a63;--muted:#59799f;--border:#c6ddf6}
*{box-sizing:border-box;margin:0;padding:0}
body{background:radial-gradient(circle at 10% 0%, rgba(255,173,89,.25) 0%, rgba(255,173,89,0) 30%),radial-gradient(circle at 85% 4%, rgba(47,125,225,.18) 0%, rgba(47,125,225,0) 36%),linear-gradient(180deg,#ffffff,#f3f9ff);color:var(--text);font-family:\"Segoe UI\",system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{background:linear-gradient(90deg,#fef7ed,#e7f2ff);border-bottom:1px solid var(--border);padding:14px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:1.2rem;font-weight:700;background:linear-gradient(90deg,var(--accent),#c97a2f);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
main{flex:1;display:flex;overflow:hidden}
.chat-col{flex:3;display:flex;flex-direction:column;padding:16px 16px 0;gap:10px;min-width:0}
#msgs{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:12px;padding-bottom:8px}
#msgs::-webkit-scrollbar{width:4px}
#msgs::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.row{display:flex;gap:8px;max-width:86%}
.row.user{align-self:flex-end;flex-direction:row-reverse}
.row.bot{align-self:flex-start}
.av{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0}
.user .av{background:linear-gradient(120deg,var(--accent),#7db8ff)}
.bot .av{background:var(--card);border:1px solid var(--border)}
.bub{padding:10px 14px;border-radius:14px;font-size:.88rem;line-height:1.55;word-break:break-word}
.user .bub{background:linear-gradient(135deg,#3f84e8,#74adff);border-radius:14px 0 14px 14px;color:#fff}
.bot .bub{background:#ffffff;border-radius:0 14px 14px 14px;border:1px solid var(--border)}
.typing{display:flex;gap:4px;padding:10px 14px}
.typing span{width:7px;height:7px;background:var(--accent);border-radius:50%;animation:dot 1.2s infinite}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes dot{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-7px)}}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:#f6fbff;border:1px solid var(--border);border-radius:20px;padding:5px 12px;font-size:.76rem;color:var(--muted);cursor:pointer;transition:all .2s;white-space:nowrap}
.chip:hover{border-color:var(--accent);color:#305c8d;background:#fff}
.input-row{display:flex;gap:8px;padding:10px 0 14px}
#inp{flex:1;background:#fff;border:1px solid var(--border);border-radius:10px;padding:10px 14px;color:var(--text);font-size:.88rem;outline:none;resize:none;height:44px;transition:border-color .2s}
#inp:focus{border-color:var(--accent)}
#snd{background:var(--accent);border:none;border-radius:10px;width:44px;height:44px;color:#fff;font-size:1.15rem;cursor:pointer;flex-shrink:0;transition:opacity .2s}
#snd:hover{opacity:.85}
#snd:disabled{opacity:.35;cursor:not-allowed}
#rst{background:#fff;border:1px solid var(--border);border-radius:10px;padding:0 14px;height:44px;color:var(--muted);font-size:.78rem;cursor:pointer;white-space:nowrap;transition:border-color .2s}
#rst:hover{border-color:var(--accent2);color:#b5681e}
.sidebar{flex:2;background:linear-gradient(180deg,#f9fcff,#eef5ff);border-left:1px solid var(--border);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;min-width:240px;max-width:360px}
.sidebar h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600;padding-bottom:6px;border-bottom:1px solid var(--border)}
.pc{background:linear-gradient(180deg,#ffffff,#edf6ff);border:1px solid var(--border);border-radius:12px;padding:12px 14px;transition:border-color .2s,transform .2s;cursor:pointer;text-align:left;color:var(--text)}
.pc:hover{transform:translateY(-1px);border-color:var(--accent2)}
.pc:first-of-type{border-color:var(--accent)}
.pc .rnk{font-size:.68rem;color:#c07a31;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.pc .br{font-size:.93rem;font-weight:700}
.pc .ct{font-size:.73rem;color:var(--muted);margin-top:2px}
.pc .pr{font-size:1.08rem;font-weight:800;color:#b66a1b;margin-top:6px}
.pc .id{font-size:.67rem;color:#5a7ea8;margin-top:4px;font-family:monospace}
.pc .rt{font-size:.82rem;font-weight:900;color:#153c69;letter-spacing:.02em}
.empty{color:var(--muted);font-size:.83rem;text-align:center;margin-top:20px;line-height:1.65}
@media(max-width:600px){.sidebar{display:none}}
</style>
</head>
<body>
<header>
  <div class=\"dot\"></div>
  <div><h1>E-Commerce AI Shopping Assistant</h1></div>
</header>
<main>
  <div class=\"chat-col\">
    <div id=\"msgs\">
      <div class=\"row bot\"><div class=\"av\">🤖</div><div class=\"bub\">Hi! Tell me what you're looking for and I'll find the best products from the catalog. Mention brand, budget, category, or just describe what you need!</div></div>
    </div>
    <div class=\"chips\" id=\"chips\">
      <span class=\"chip\">Good after-shave with nice smell</span>
      <span class=\"chip\">Organic skin care under ₹200</span>
      <span class=\"chip\">Anti-bacterial hand wash under ₹100</span>
      <span class=\"chip\">Premium green tea</span>
      <span class=\"chip\">Same thing but cheaper</span>
      <span class=\"chip\">What's in it? Any alcohol?</span>
    </div>
    <div class=\"input-row\">
      <textarea id=\"inp\" placeholder=\"Describe what you're looking for...\" rows=\"1\"></textarea>
      <button id=\"snd\" type=\"button\" title=\"Send\">&#9658;</button>
      <button id=\"rst\" type=\"button\">Reset</button>
    </div>
  </div>
  <aside class=\"sidebar\">
    <h2>Recommendations</h2>
    <div id=\"prods\"><p class=\"empty\">Products will appear here after your first message.</p></div>
  </aside>
</main>
<script>
const msgs=document.getElementById('msgs'),inp=document.getElementById('inp'),snd=document.getElementById('snd'),rst=document.getElementById('rst'),prods=document.getElementById('prods');
let latestProducts=[];
function scroll(){msgs.scrollTop=msgs.scrollHeight}
function addBubble(role,text){const r=document.createElement('div');r.className='row '+role;const a=document.createElement('div');a.className='av';a.textContent=role==='user'?'👤':'🤖';const b=document.createElement('div');b.className='bub';b.textContent=text;r.appendChild(a);r.appendChild(b);msgs.appendChild(r);scroll();return b}
function showTyping(){const r=document.createElement('div');r.className='row bot';r.id='typ';r.innerHTML='<div class="av">🤖</div><div class="bub"><div class="typing"><span></span><span></span><span></span></div></div>';msgs.appendChild(r);scroll()}
function removeTyping(){const e=document.getElementById('typ');if(e)e.remove()}
function ingredientLine(desc){if(!desc)return '';const lower=desc.toLowerCase();const idx=lower.indexOf('ingredients');if(idx<0)return '';let tail=desc.slice(idx);let sep=tail.indexOf(':');if(sep<0)sep=tail.indexOf('-');if(sep<0)return '';tail=tail.slice(sep+1).trim();const end=tail.indexOf('.');const out=end>=0?tail.slice(0,end):tail;return out?('Ingredients: '+out.trim()):''}
async function selectProductOnServer(productId){if(!productId)return;try{await fetch('/select-product',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product_id:String(productId)})})}catch(e){}}
function showProductDetail(idx){const p=latestProducts[idx];if(!p)return;const nm=p.product_name||'Unknown product';const br=p.brand||'N/A';const pr=(p.sale_price||p.sale_price===0)?('₹'+parseFloat(p.sale_price).toFixed(2)):'N/A';const rt=(p.rating||p.rating===0)?p.rating:'N/A';const desc=(p.description||'No detailed description available in catalog.').trim();const ing=ingredientLine(desc);const detail='Product details: '+nm+' | Brand: '+br+' | Price: '+pr+' | Rating: '+rt+'. '+(ing?ing+'. ':'')+'Description: '+desc;addBubble('bot',detail)}
function renderProducts(ps){latestProducts=ps||[];if(!latestProducts.length){prods.innerHTML='<p class="empty">No products found. Try a more specific query!</p>';return}prods.innerHTML='';latestProducts.forEach(function(p,i){const nm=p.product_name||'Unknown product';let br=p.brand||'Generic';br=br.charAt(0).toUpperCase()+br.slice(1);const ct=(p.category||'N/A')+' > '+(p.sub_category||'N/A');const pr=(p.sale_price||p.sale_price===0)?'₹'+parseFloat(p.sale_price).toFixed(2):'N/A';const rt=(p.rating||p.rating===0)?p.rating:'N/A';const rnk=i===0?'Top Pick':'#'+(i+1);const c=document.createElement('button');c.type='button';c.className='pc';c.innerHTML='<div class="rnk">'+rnk+'</div><div class="br">'+nm+'</div><div class="ct">'+br+' • '+ct+'</div><div class="pr">'+pr+'</div><div class="id"><span class="rt">Rating: '+rt+'</span> | ID: '+(p.product_id||'N/A')+'</div>';c.addEventListener('click',async function(){await selectProductOnServer(p.product_id);showProductDetail(i)});prods.appendChild(c)})}
async function send(text){text=(text||'').trim();if(!text)return;if(snd.disabled)return;inp.value='';document.getElementById('chips').style.display='none';addBubble('user',text);snd.disabled=true;inp.disabled=true;showTyping();try{const ctrl=new AbortController();const tid=setTimeout(function(){ctrl.abort()},60000);const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text}),signal:ctrl.signal});clearTimeout(tid);const d=await r.json();removeTyping();if(d.error){addBubble('bot','Error: '+d.error)}else{addBubble('bot',d.reply);renderProducts(d.products)}}catch(e){removeTyping();addBubble('bot',e.name==='AbortError'?'Request timed out. The server may be busy — please try again.':'Network error — server may have restarted. Refresh the page.')}finally{snd.disabled=false;inp.disabled=false;inp.focus()}}
async function resetChat(){await fetch('/reset',{method:'POST'});msgs.innerHTML='';addBubble('bot','Conversation reset! What can I help you find today?');prods.innerHTML='<p class="empty">Products will appear here after your first message.</p>';latestProducts=[];document.getElementById('chips').style.display='flex'}
snd.addEventListener('click',()=>send(inp.value));
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send(inp.value)}});
document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',()=>send(c.textContent||'')));
rst.addEventListener('click',resetChat);
inp.focus();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        pipeline = get_pipeline()
        reply = pipeline.chat(message)
        products = pipeline.last_products
        return jsonify({"reply": reply, "products": products})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    get_pipeline().reset()
    return jsonify({"ok": True})


@app.route("/select-product", methods=["POST"])
def select_product():
  data = request.get_json(force=True)
  product_id = (data.get("product_id") or "").strip()
  if not product_id:
    return jsonify({"error": "Empty product_id"}), 400
  get_pipeline().select_product(product_id)
  return jsonify({"ok": True})


if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\n GROQ_API_KEY not set. export GROQ_API_KEY=YOUR_GROQ_KEY\n")
        sys.exit(1)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("Loading CRS pipeline ...")
    pipeline = get_pipeline()

    print("Warming up embedding model ...")
    pipeline.retriever.search("test warm up query", top_k=1)
    print("Ready! Open http://localhost:7860")

    app.run(host="0.0.0.0", port=7860, debug=False, threaded=False)
