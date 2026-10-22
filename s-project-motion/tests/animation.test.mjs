import assert from 'node:assert/strict';
import path from 'node:path';
import {mkdir, readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {launchBrowser} from '../scripts/browser.mjs';
import {startServer} from '../scripts/server.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const browser = await launchBrowser();
const server = await startServer();
const url = server.url + '/';
let count=0;
const check = async (name,run) => {await run(); count++; console.log(`PASS ${name}`);};
try {
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  const errors=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url+'?render=1');
  await page.waitForFunction(()=>Boolean(window.logoMotion));
  await check('10 asymmetric rays and 10 endpoint circles; original coordinates and common origin',async()=>{
    const result=await page.evaluate(()=>({
      rays:[...document.querySelectorAll('[data-ray]')].map(p=>p.getAttribute('d')),
      dots:[...document.querySelectorAll('[data-dot]')].map(c=>[+c.getAttribute('cx'),+c.getAttribute('cy')]),
    }));
    const expected=[[278,85],[210,102],[126,147],[71,204],[165,220],[152,267],[204,303],[252,314],[246,225],[250,183]];
    assert.deepEqual(result.dots,expected);
    assert.deepEqual(result.rays,expected.map(([x,y])=>`M203 205 L${x} ${y}`));
  });
  const inspect=async t=>page.evaluate(t=>{
    window.logoMotion.seek(t);
    const attrs=s=>[...document.querySelectorAll(s)].map(e=>({opacity:+e.getAttribute('opacity'),offset:+e.getAttribute('stroke-dashoffset'),transform:e.getAttribute('transform')}));
    return {center:attrs('[data-center]')[0],lines:attrs('[data-ray]'),dots:attrs('[data-dot]'),text:attrs('[data-wordmark]')[0]};
  },t);
  await check('five phases: empty start, center, rays, endpoints, wordmark, static finish',async()=>{
    let s=await inspect(0);
    assert.equal(s.center.opacity,0); assert(s.lines.every(x=>x.opacity===0)); assert(s.dots.every(x=>x.opacity===0)); assert.equal(s.text.opacity,0);
    s=await inspect(.2); assert.equal(s.center.opacity,1); assert(s.lines.every(x=>x.opacity===0));
    s=await inspect(.65); assert(s.lines.some(x=>x.offset>0&&x.offset<1)); assert(s.dots.every(x=>x.opacity===0)); assert.equal(s.text.opacity,0);
    s=await inspect(1.1); assert(s.lines.every(x=>x.offset<1e-8)); assert.equal(s.text.opacity,0);
    s=await inspect(1.4); assert(s.dots.every(x=>x.opacity===1)); assert.equal(s.text.opacity,0);
    s=await inspect(1.75); assert(s.text.opacity>0&&s.text.opacity<1);
    const end=await inspect(2.1); assert.equal(end.text.opacity,1); assert.equal(end.text.transform,'translate(0 0)');
    assert.deepEqual(await inspect(2.5),end);
  });
  await check('rays grow monotonically; endpoint scales and opacity never overshoot',async()=>{
    let previous=Array(10).fill(1);
    for(let t=.2;t<=1.1;t+=.05){
      const state=await inspect(t);
      state.lines.forEach((l,i)=>{assert(l.offset<=previous[i]+1e-9);assert(l.offset>=0&&l.offset<=1);});
      previous=state.lines.map(x=>x.offset);
    }
    for(let t=1.1;t<=1.4;t+=.03){
      const state=await inspect(t);
      state.dots.forEach(d=>{assert(d.opacity>=0&&d.opacity<=1);const scale=+d.transform.match(/scale\(([^)]+)/)[1];assert(scale>=.85&&scale<=1);});
    }
  });
  await check('final frame remains identical after holding; seek is deterministic',async()=>{
    const first=await inspect(2.5); await inspect(.73); assert.deepEqual(await inspect(2.5),first);
    const a=await page.locator('#logo').screenshot(); await page.waitForTimeout(200);
    assert.deepEqual(await page.locator('#logo').screenshot(),a);
  });
  await page.goto(url);
  await check('autoplay once, then stop at 2.5 seconds',async()=>{
    await page.waitForFunction(()=>window.logoMotion?.getState().time>.03);
    assert.equal(await page.evaluate(()=>window.logoMotion.getState().playing),true);
    await page.waitForFunction(()=>!window.logoMotion.getState().playing,{},{timeout:5000});
    const state=await page.evaluate(()=>window.logoMotion.getState());
    assert.equal(state.time,2.5);assert.equal(state.loop,false);
  });
  await check('Replay, pause, reset and timeline operate correctly',async()=>{
    await page.click('#replay');
    await page.waitForFunction(()=>window.logoMotion.getState().time>.05);
    await page.click('#toggle');
    const paused=await page.evaluate(()=>window.logoMotion.getState().time);
    await page.waitForTimeout(150);assert.equal(await page.evaluate(()=>window.logoMotion.getState().time),paused);
    await page.click('#reset');assert.equal(await page.evaluate(()=>window.logoMotion.getState().time),0);
    await page.locator('#timeline').evaluate(input=>{input.value='1.75';input.dispatchEvent(new Event('input',{bubbles:true}));});
    const s=await page.evaluate(()=>window.logoMotion.getState());assert.equal(s.time,1.75);assert.equal(s.playing,false);
  });
  await check('speed changes elapsed time and loop restarts',async()=>{
    await page.selectOption('#speed','2');
    await page.click('#replay');
    await page.waitForTimeout(210);
    const s=await page.evaluate(()=>window.logoMotion.getState());assert.equal(s.speed,2);assert(s.time>.3&&s.time<.8);
    await page.click('#toggle');await page.check('#loop');
    await page.evaluate(()=>window.logoMotion.seek(2.48));await page.click('#toggle');
    await page.waitForTimeout(160);
    const loop=await page.evaluate(()=>window.logoMotion.getState());assert(loop.playing&&loop.loop&&loop.time<1);
    await page.click('#toggle');await page.uncheck('#loop');await page.selectOption('#speed','1');
  });
  await check('three backgrounds: white ink on dark, genuine transparency',async()=>{
    await page.selectOption('#background','dark');
    assert.equal(await page.locator('#logo').getAttribute('color'),'#f7f7f7');
    await page.selectOption('#background','transparent');
    assert.equal(await page.locator('[data-background-rect]').getAttribute('opacity'),'0');
    await page.selectOption('#background','light');
    assert.equal(await page.locator('[data-background-rect]').getAttribute('opacity'),'1');
  });
  await check('transparent capture has alpha=0 in background and alpha=255 in the logo',async()=>{
    const capture=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
    try {
      await capture.goto(url+'?render=1&background=transparent');
      await capture.evaluate(()=>window.logoMotion.seek(2.5));
      const bytes=await capture.screenshot({omitBackground:true});
      const alpha=await capture.evaluate(async bytes=>{
        const bitmap=await createImageBitmap(new Blob([new Uint8Array(bytes)],{type:'image/png'}));
        const canvas=document.createElement('canvas');canvas.width=1920;canvas.height=1080;
        const ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);
        return [ctx.getImageData(0,0,1,1).data[3],ctx.getImageData(590,553,1,1).data[3]];
      },Array.from(bytes));
      assert.deepEqual(alpha,[0,255]);
    } finally {await capture.close();}
  });
  await check('PNG and SVG buttons create real final-frame downloads',async()=>{
    await page.locator('#timeline').evaluate(input=>{input.value='.1';input.dispatchEvent(new Event('input',{bubbles:true}));});
    const svgPromise=page.waitForEvent('download');await page.click('#save-svg');const svg=await svgPromise;
    assert.equal(svg.suggestedFilename(),'s-project-light.svg');
    const source=await readFile(await svg.path(),'utf8');
    assert.equal((source.match(/stroke-dashoffset="0"/g)||[]).length,10);
    assert(source.includes('width="1920"'));assert(!source.includes('opacity="0"'));
    const pngPromise=page.waitForEvent('download');await page.click('#save-png');const png=await pngPromise;
    const bytes=await readFile(await png.path());assert.equal(bytes.subarray(1,4).toString(),'PNG');
    assert.equal(bytes.readUInt32BE(16),1920);assert.equal(bytes.readUInt32BE(20),1080);
  });
  await check('responsive layout and full logo bounds at 375, 768, 1920 pixels',async()=>{
    await page.evaluate(()=>{window.logoMotion.pause();window.logoMotion.seek(2.5);});
    for(const width of [375,768,1920]){
      await page.setViewportSize({width,height:1080});
      const b=await page.evaluate(()=>{
        const svg=document.querySelector('#logo').getBoundingClientRect();
        const mark=document.querySelector('[data-composition]').getBoundingClientRect();
        const text=document.querySelector('[data-wordmark]').getBoundingClientRect();
        const controls=document.querySelector('.transport').getBoundingClientRect();
        return {svg:svg.toJSON(),mark:mark.toJSON(),text:text.toJSON(),controls:controls.toJSON(),overflow:document.documentElement.scrollWidth>innerWidth};
      });
      assert(!b.overflow,`Horizontal overflow at ${width}`);
      assert(b.mark.left>b.svg.left&&b.mark.right<b.svg.right&&b.mark.top>b.svg.top&&b.mark.bottom<b.svg.bottom);
      assert(b.controls.top>=b.svg.bottom);
      const target=b.text.x;
      await page.evaluate(()=>window.logoMotion.seek(2.1));
      assert.equal((await page.locator('[data-wordmark]').boundingBox()).x,target);
    }
  });
  await mkdir(path.join(root,'test-results'),{recursive:true});
  await page.setViewportSize({width:1440,height:1100});
  await page.screenshot({path:path.join(root,'test-results/interface.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  console.log(`\n${count} checks passed. No browser errors.`);
} finally {await browser.close();await server.close();}
