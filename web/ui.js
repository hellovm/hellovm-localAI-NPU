export async function openWizard(ctx, onDone){
  const ov=document.querySelector('#wizard_overlay');
  const w=document.querySelector('#wizard');
  w.innerHTML='';
  const step=document.createElement('div');
  step.className='wizard-step';
  const title=document.createElement('div');
  title.textContent='配置向导';
  const s1=document.createElement('div');
  const lab1=document.createElement('label');
  lab1.textContent='性能模式';
  const sel1=document.createElement('select');
  ['LATENCY','THROUGHPUT'].forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel1.appendChild(o)});
  sel1.value=ctx.perf||'LATENCY';
  const s2=document.createElement('div');
  const lab2=document.createElement('label');
  lab2.textContent='NPU并行度';
  const inp=document.createElement('input');
  inp.type='number';inp.min='1';inp.value=String(ctx.streams||1);
  const row=document.createElement('div');
  const ok=document.createElement('button');
  ok.textContent='应用';
  ok.onclick=()=>{ov.classList.add('hidden');onDone&&onDone({perf:sel1.value,streams:parseInt(inp.value,10)||1})};
  const cancel=document.createElement('button');
  cancel.textContent='取消';
  cancel.onclick=()=>{ov.classList.add('hidden')};
  s1.appendChild(lab1);s1.appendChild(sel1);
  s2.appendChild(lab2);s2.appendChild(inp);
  row.appendChild(ok);row.appendChild(cancel);
  step.appendChild(title);step.appendChild(s1);step.appendChild(s2);step.appendChild(row);
  w.appendChild(step);
  ov.classList.remove('hidden');
}

export function attachTooltips(){
  const tips=document.querySelectorAll('.help');
  tips.forEach(el=>{
    let t;
    function show(){
      if(t)return; t=document.createElement('div');
      t.className='tooltip show-tooltip';
      t.textContent=el.getAttribute('data-tooltip')||'';
      el.appendChild(t);
    }
    function hide(){
      if(t){t.remove();t=null}
    }
    el.addEventListener('mouseenter',show);
    el.addEventListener('mouseleave',hide);
    el.addEventListener('focus',show);
    el.addEventListener('blur',hide);
  })
}