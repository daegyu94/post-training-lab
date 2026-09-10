/* Read-only run archives. All history values come from the selected saved record. */
(() => {
  const host = document.getElementById('run-history');
  const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (value, digits=1) => Number.isFinite(value) ? value.toLocaleString(undefined,{maximumFractionDigits:digits}) : '—';
  const date = value => value ? new Date(value).toLocaleString() : '—';
  const duration = value => value == null ? '진행 중 snapshot' : value < 60 ? number(value)+' s' : number(value/60)+' min';
  const badge = (label, cls='') => `<span class="history-badge ${cls}">${esc(label)}</span>`;
  const source = m => badge(m.synthetic?'Synthetic example':'Measured',m.synthetic?'synthetic':'completed');
  const statuses = {running:'Running snapshot',completed:'Completed',failed:'Failed',cancelled:'Cancelled',captured:'Captured'};
  const status = m => badge(statuses[m.status] || m.status,m.status);
  let registry=null, record=null, request=0, selected=new Set(), tab='overview';
  let filters={search:'',status:'all',source:'all'};
  async function json(path) {
    const response=await fetch(path);
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
  function route(id) {
    const url=new URL(location.href);
    if(id)url.searchParams.set('archive',id);else url.searchParams.delete('archive');
    url.hash='history';history.pushState(null,'',url);
    show();
  }
  function error(message) {
    host.innerHTML=`<div class="panel history-empty"><h2>기록을 불러오지 못했습니다</h2><p class="history-error">${esc(message)}</p><button class="button" id="history-retry">다시 시도</button> <button class="button" id="history-home">실행 목록</button></div>`;
    host.querySelector('#history-retry').onclick=show;
    host.querySelector('#history-home').onclick=()=>route(null);
  }
  async function show() {
    const token=++request;
    host.innerHTML='<div class="history-empty" role="status">실행 이력을 불러오는 중…</div>';
    try {
      if(!registry)registry=await json('api/history/index.json');
      if(token!==request)return;
      const id=new URL(location.href).searchParams.get('archive');
      if(!id){record=null;list();return;}
      if(!registry.runs.some(r=>r.archive_id===id))throw new Error('등록되지 않은 archive ID입니다.');
      const next=await json('api/history/'+encodeURIComponent(id)+'.json');
      if(token!==request)return;
      record=next;tab='overview';detail();
    } catch(e) {if(token===request)error(e.message);}
  }
  function list() {
    const runs=registry.runs;
    host.innerHTML=`<div class="history-heading"><div><h2>Run history</h2><p>실행이 끝난 뒤에도 설정과 관측 기록을 다시 확인하세요. 저장된 snapshot을 조회하며, 실행 중인 작업의 실시간 상태는 아닙니다.</p></div></div>
    <div class="kpis history-stats">${[['Saved records',runs.length],['Completed training',runs.filter(r=>r.kind==='training'&&r.status==='completed').length],['Measured records',runs.filter(r=>!r.synthetic).length],['Synthetic examples',runs.filter(r=>r.synthetic).length]].map(([label,value])=>`<article class="card kpi"><span class="eyebrow">${label}</span><div class="kpi-value">${value}</div></article>`).join('')}</div>
    <div class="history-tools"><label class="search-field">Search runs<input id="history-search" type="search" placeholder="이름, run ID, framework 검색" value="${esc(filters.search)}"></label>
    <label>Status<select id="history-status"><option value="all">All statuses</option>${Object.entries(statuses).map(([key,value])=>`<option value="${key}">${value}</option>`).join('')}</select></label>
    <label>Data source<select id="history-source"><option value="all">All sources</option><option value="measured">Measured</option><option value="synthetic">Synthetic</option></select></label>
    <button class="button" id="history-compare">Compare (0/2)</button></div><div class="panel table-wrap"><table class="table history-table"><thead><tr><th>Compare</th><th>Run</th><th>Status / Source</th><th>Started / Ended</th><th>Duration</th><th>Resources</th></tr></thead><tbody id="history-rows"></tbody></table><div id="history-empty" class="history-empty" hidden>조건에 맞는 기록이 없습니다.</div></div><p id="history-count" class="history-count"></p><div id="history-comparison"></div>`;
    host.querySelector('#history-status').value=filters.status;
    host.querySelector('#history-source').value=filters.source;
    for(const [id,key] of [['history-search','search'],['history-status','status'],['history-source','source']])host.querySelector('#'+id).addEventListener(key==='search'?'input':'change',e=>{filters[key]=e.target.value;rows();});
    host.querySelector('#history-compare').onclick=compare;
    rows();
  }
  function rows() {
    const visible=registry.runs.filter(r=>(filters.status==='all'||r.status===filters.status)&&(filters.source==='all'||r.synthetic===(filters.source==='synthetic'))&&[r.run_id,r.name,r.framework].join(' ').toLowerCase().includes(filters.search.toLowerCase()));
    host.querySelector('#history-rows').innerHTML=visible.map(m=>`<tr><td><input type="checkbox" data-compare="${esc(m.archive_id)}" aria-label="Compare ${esc(m.name)}" ${selected.has(m.archive_id)?'checked':''}></td><td><button class="run-link" data-open="${esc(m.archive_id)}">${esc(m.name)}</button><small>${esc(m.archive_id)} · ${esc(m.framework)}</small><small>${esc(m.workload)}</small></td><td>${status(m)}<small>${source(m)}</small></td><td><time>${esc(date(m.started_at))}</time><small>${m.ended_at?esc(date(m.ended_at)):'종료 기록 없음'}</small></td><td>${esc(duration(m.duration_seconds))}</td><td>${esc(m.nodes)} nodes<small>${m.gpus==null?'GPU 미수집':esc(m.gpus)+' GPUs'}</small></td></tr>`).join('');
    host.querySelector('#history-empty').hidden=visible.length>0;
    host.querySelector('#history-count').textContent=`${visible.length} / ${registry.runs.length} records · 시각은 브라우저 현지 시간`;
    host.querySelectorAll('[data-open]').forEach(button=>button.onclick=()=>route(button.dataset.open));
    host.querySelectorAll('[data-compare]').forEach(input=>input.onchange=()=>{if(input.checked&&selected.size<2)selected.add(input.dataset.compare);else selected.delete(input.dataset.compare);rows();});
    host.querySelectorAll('[data-compare]').forEach(input=>input.disabled=selected.size===2&&!input.checked);
    const button=host.querySelector('#history-compare');button.textContent=`Compare (${selected.size}/2)`;button.disabled=selected.size!==2;
  }
  const fields = pairs => `<dl>${pairs.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl>`;
  function detail() {
    const m=record.metadata,full=record.report;
    const canProfile=['summary','resources','storage','interconnect','data_movement'].every(key=>full[key])&&full.summary.diagnostic&&Array.isArray(full.resources.nodes)&&Array.isArray(full.storage.series)&&Array.isArray(full.data_movement.paths);
    host.innerHTML=`<button class="button history-back" id="history-back">← All runs</button><div class="history-heading"><div><h2>${esc(m.name)}</h2><p>${esc(m.archive_id)} · ${esc(m.framework)} · ${esc(m.workload)}</p></div><div class="history-actions">${status(m)}${source(m)}<button class="button" id="history-download">Export archive</button>${canProfile?'<button class="button primary" id="history-profile">Profiling details</button>':''}</div></div>
    <div class="history-note">${m.kind==='host'?'노드 전체 CPU·메모리·NIC 관측 기록입니다. 학습 작업의 성공 여부와 GPU·rank 지표는 수집하지 않았습니다.':'이 실행의 저장된 기록입니다. 재조회 시 수치를 생성하거나 변경하지 않습니다.'}${m.synthetic?' 합성 예제이며 실제 학습 결과가 아닙니다.':''}</div>
    <div class="history-tabs" aria-label="Run detail sections">${[['overview','Overview'],['timeline','Timeline'],['resources','Resources'],['configuration','Configuration']].map(([key,label])=>`<button class="button" data-history-tab="${key}" aria-pressed="${tab===key}">${label}</button>`).join('')}</div><div id="history-detail"></div>`;
    host.querySelector('#history-back').onclick=()=>route(null);
    host.querySelector('#history-download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=m.archive_id+'-archive.json';a.click();URL.revokeObjectURL(url);};
    if(canProfile)host.querySelector('#history-profile').onclick=()=>window.dispatchEvent(new CustomEvent('history-open-profile',{detail:record}));
    host.querySelectorAll('[data-history-tab]').forEach(button=>button.onclick=()=>{tab=button.dataset.historyTab;detail();});
    const area=host.querySelector('#history-detail');
    if(tab==='overview') {
      const metrics=full.summary?.metrics;
      area.innerHTML=`<div class="history-detail-grid"><article class="panel history-details"><h3>Execution</h3>${fields([['Run ID',m.run_id],['Started',date(m.started_at)],['Ended',date(m.ended_at)],['Duration',duration(m.duration_seconds)],['Saved samples',m.sample_count],['Archive created',date(m.archived_at)]])}</article><article class="panel history-details"><h3>${metrics?'Final snapshot':'Observation scope'}</h3>${metrics?fields([['Throughput',number(metrics.training_tokens_per_second,0)+' tokens/s'],['Step P95',number(metrics.training_step_time_seconds_p95)+' s'],['GPU utilization',number(metrics.gpu_utilization_percent)+'%'],['Training loss',number(metrics.training_loss,3)],['Nodes / GPUs',`${m.nodes} / ${m.gpus}`]]):fields([['Scope','Node-wide, not training-process attributed'],['Nodes',m.nodes],['GPU / rank','Not collected'],['Outcome','Observation captured; training outcome unknown']])}</article></div>${m.failure_reason?`<div class="history-note history-error"><strong>Failure reason</strong><p>${esc(m.failure_reason)}</p></div>`:''}<article class="panel history-details"><h3>Recorded events</h3>${events()}</article>`;
    } else if(tab==='configuration') {
      area.innerHTML='<article class="panel history-details"><h3>Saved configuration</h3><pre class="history-json" id="history-config"></pre></article>';
      area.querySelector('pre').textContent=record.history.configuration?JSON.stringify(record.history.configuration,null,2):'학습 설정을 수집하지 않은 노드 관측 기록입니다.';
    } else if(tab==='resources') resources(area);
    else timeline(area);
  }
  function events() {
    const items=record.history.events||[];
    if(!items.length)return '<p class="history-empty">저장된 실행 이벤트가 없습니다.</p>';
    return `<ol class="history-events">${items.map(e=>`<li class="${e.level==='warning'?'warning':''}"><time>${esc(date(e.timestamp))}</time><strong>${esc(e.title)}</strong><p>${esc(e.message)}</p></li>`).join('')}</ol>`;
  }
  function resources(area) {
    let nodes=record.report.resources?.nodes;
    if(record.metadata.kind==='host') {
      const latest=new Map();for(const sample of record.history.samples)latest.set(sample.node,sample);
      nodes=Array.from(latest.values()).map(s=>({node:s.node,...s.metrics,nic_transmit_gbps:s.metrics.nic_transmit_gbps}));
    }
    if(!nodes?.length){area.innerHTML='<div class="history-empty">저장된 리소스 정보가 없습니다.</div>';return;}
    area.innerHTML=`<article class="panel"><div class="panel-head"><div><h3>Node resources · saved snapshot</h3><p>각 노드의 마지막 저장값 · GPU 평균은 수집된 경우에만 표시합니다</p></div></div><div class="table-wrap"><table class="table"><thead><tr><th>Node</th><th>CPU</th><th>Memory</th><th>NIC TX</th><th>GPU utilization</th></tr></thead><tbody>${nodes.map(n=>`<tr><td>${esc(n.node)}</td><td>${number(n.cpu_utilization_percent)}%</td><td>${number(n.memory_used_gib)} / ${number(n.memory_total_gib)} GiB</td><td>${number(n.nic_transmit_gbps,3)} Gbps</td><td>${n.gpus?.length?number(n.gpus.reduce((s,g)=>s+g.utilization_percent,0)/n.gpus.length)+'%':'미수집'}</td></tr>`).join('')}</tbody></table></div></article>`;
  }
  function timeline(area) {
    const samples=record.history.samples||[],hostRun=record.metadata.kind==='host';
    if(!samples.length){area.innerHTML='<div class="panel history-empty">저장된 시계열이 없습니다. Snapshot 요약만 보관된 실행입니다.</div>';return;}
    const options=hostRun?[['cpu_utilization_percent','CPU · %'],['memory_used_gib','Memory · GiB'],['nic_transmit_gbps','NIC TX · Gbps']]:[['training_tokens_per_second','Throughput · tokens/s'],['training_step_time_seconds_p95','Step P95 · s'],['gpu_utilization_percent','GPU utilization · %'],['training_loss','Training loss']];
    area.innerHTML=`<article class="panel history-details"><div class="history-tools"><label>Metric<select id="history-metric">${options.map(([key,label])=>`<option value="${key}">${label}</option>`).join('')}</select></label>${hostRun?`<label>Node<select id="history-node">${Array.from(new Set(samples.map(s=>s.node))).map(n=>`<option>${esc(n)}</option>`).join('')}</select></label>`:''}</div><svg class="history-chart" id="history-chart" role="img" aria-label="Saved run metric timeline"></svg><div class="history-chart-caption" id="history-chart-caption"></div></article><article class="panel history-details" style="margin-top:20px"><h3>Events</h3>${events()}</article>`;
    area.querySelectorAll('select').forEach(select=>select.onchange=chart);chart();
  }
  function chart() {
    const svg=host.querySelector('#history-chart');if(!svg||!record)return;
    const key=host.querySelector('#history-metric').value,node=host.querySelector('#history-node')?.value;
    const data=(record.history.samples||[]).filter(s=>(!node||s.node===node)&&Number.isFinite(s.metrics[key])).map(s=>({time:new Date(s.timestamp||s.received_at).getTime(),value:s.metrics[key]})).sort((a,b)=>a.time-b.time);
    if(!data.length){svg.innerHTML='';host.querySelector('#history-chart-caption').textContent='이 metric의 저장된 샘플이 없습니다.';return;}
    const width=Math.max(300,svg.clientWidth),min=Math.min(...data.map(d=>d.value)),max=Math.max(...data.map(d=>d.value)),pad=Math.max((max-min)*.15,Math.abs(max)*.03,.01),low=Math.max(0,min-pad),high=max+pad;
    const start=data[0].time,end=data.at(-1).time;
    svg.setAttribute('viewBox',`0 0 ${width} 230`);
    svg.innerHTML=Array.from({length:4},(_,i)=>{const y=20+i*55;return `<line x1="65" y1="${y}" x2="${width-10}" y2="${y}"/><text x="0" y="${y+5}">${esc((high-(high-low)*i/3)>=1000?number((high-(high-low)*i/3)/1000,1)+'k':number(high-(high-low)*i/3,2))}</text>`;}).join('')+`<polyline points="${data.map(d=>`${65+(d.time-start)/Math.max(1,end-start)*(width-75)},${185-(d.value-low)/(high-low)*165}`).join(' ')}"/>`;
    if(data.length===1)svg.innerHTML+=`<circle cx="65" cy="${185-(data[0].value-low)/(high-low)*165}" r="4" fill="var(--accent)"/>`;
    host.querySelector('#history-chart-caption').textContent=`${date(new Date(start).toISOString())} → ${date(new Date(end).toISOString())} · ${data.length} saved samples`;
  }
  async function compare() {
    const area=host.querySelector('#history-comparison'),ids=Array.from(selected);area.textContent='비교 기록을 불러오는 중…';
    try {
      const docs=await Promise.all(ids.map(id=>json('api/history/'+encodeURIComponent(id)+'.json')));
      if(!area.isConnected)return;
      const rows=[['Status',d=>statuses[d.metadata.status]],['Data source',d=>d.metadata.synthetic?'Synthetic':'Measured'],['Framework',d=>d.metadata.framework],['Workload',d=>d.metadata.workload],['Duration',d=>duration(d.metadata.duration_seconds)],['Nodes',d=>d.metadata.nodes],['GPUs',d=>d.metadata.gpus],['Throughput · tokens/s',d=>number(d.report.summary?.metrics.training_tokens_per_second,0)],['Step P95 · s',d=>number(d.report.summary?.metrics.training_step_time_seconds_p95)],['GPU utilization · %',d=>number(d.report.summary?.metrics.gpu_utilization_percent)],['Loss',d=>number(d.report.summary?.metrics.training_loss,3)],['Configuration',d=>d.history.configuration?JSON.stringify(d.history.configuration):'미수집']];
      area.innerHTML=`<article class="panel history-details" style="margin-top:20px"><h3>Run comparison</h3><p class="history-note">서로 다른 workload·모델·측정 범위의 수치는 직접적인 성능 우열을 뜻하지 않습니다. —는 미수집 항목입니다.</p><div class="table-wrap"><table class="table"><thead><tr><th>Field</th>${docs.map(d=>`<th>${esc(d.metadata.name)}</th>`).join('')}</tr></thead><tbody>${rows.map(([name,get])=>`<tr><th>${esc(name)}</th>${docs.map(d=>`<td style="white-space:normal;min-width:180px;overflow-wrap:anywhere">${esc(get(d))}</td>`).join('')}</tr>`).join('')}</tbody></table></div></article>`;
    }catch(e){if(area.isConnected)area.textContent='비교 실패: '+e.message;}
  }
  window.addEventListener('history-view',show);
  window.addEventListener('popstate',()=>{if(location.hash==='#history'||!location.hash)window.dispatchEvent(new Event('history-navigate'));});
  window.addEventListener('resize',chart);
})();
