import path from 'node:path';
import os from 'node:os';
import {mkdir, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {launchBrowser} from './browser.mjs';
import {startServer} from './server.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const options = {theme:'all', format:'all'};
for (let i=0; i<args.length; i+=2) {
  if (!['--theme','--format'].includes(args[i]) || !args[i+1]) {
    throw new Error('Usage: npm run export -- [--theme all|light|dark|transparent] [--format all|mp4|webm|png|svg]');
  }
  options[args[i].slice(2)] = args[i+1];
}
if (!['all','light','dark','transparent'].includes(options.theme) || !['all','mp4','webm','png','svg'].includes(options.format)) throw new Error('Unsupported export option');
if (options.theme === 'transparent' && options.format === 'mp4') throw new Error('MP4/H.264 has no alpha channel. Use --format webm for transparency.');
if (['all','mp4','webm'].includes(options.format) && spawnSync('ffmpeg',['-version'], {stdio:'ignore'}).status !== 0) throw new Error('FFmpeg not found. Install FFmpeg with libx264 and libvpx-vp9.');
const out = path.join(root,'exports');
await mkdir(out,{recursive:true});
const temporary = await mkdtemp(path.join(os.tmpdir(),'s-project-export-'));
const browser = await launchBrowser();
const server = await startServer();
try {
  const page = await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
  const errors=[]; page.on('pageerror', error => errors.push(error.message));
  const themes = options.theme === 'all' ? ['light','dark','transparent'] : [options.theme];
  for (const theme of themes) {
    await page.goto(server.url + `/?render=1&background=${theme}`);
    await page.waitForFunction(() => Boolean(window.logoMotion));
    await page.evaluate(() => window.logoMotion.seek(2.5));
    const suffix = theme === 'light' ? '' : `-${theme}`;
    if (['all','png'].includes(options.format)) {
      await page.screenshot({path:path.join(out,`s-project-final${suffix}.png`),omitBackground:theme==='transparent'});
    }
    if (['all','svg'].includes(options.format)) {
      const source = await page.evaluate(() => window.logoMotion.finalSvg());
      await writeFile(path.join(out,`s-project${suffix}.svg`),source);
    }
    const video = options.format === 'all' || (theme==='transparent' ? options.format==='webm' : options.format==='mp4'||options.format==='webm');
    if (video) {
      const frames = path.join(temporary,theme);
      await mkdir(frames);
      for (let frame=0;frame<75;frame++) {
        await page.evaluate(time => window.logoMotion.seek(time),frame/30);
        await page.screenshot({path:path.join(frames,`${String(frame).padStart(4,'0')}.png`),omitBackground:theme==='transparent'});
      }
      const webm = theme==='transparent'||options.format==='webm';
      const filename = path.join(out,`s-project${suffix}.${webm?'webm':'mp4'}`);
      const codec = webm
        ? ['-c:v','libvpx-vp9','-pix_fmt',theme==='transparent'?'yuva420p':'yuv420p','-auto-alt-ref','0','-b:v','0','-crf','18','-row-mt','1','-cpu-used','4']
        : ['-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-preset','medium','-movflags','+faststart'];
      const ff = spawnSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-framerate','30','-i',path.join(frames,'%04d.png'),'-frames:v','75','-an','-threads','2',...codec,filename],{stdio:'inherit'});
      if (ff.error) throw ff.error;
      if (ff.status!==0) throw new Error(`FFmpeg failed with exit code ${ff.status}`);
      console.log(`Exported ${path.relative(root,filename)}: 1920×1080, 30 fps, 75 frames, 2.5 seconds${theme==='transparent'?', alpha':''}`);
    }
    if (errors.length) throw new Error(errors.join('\n'));
  }
} finally {
  await browser.close();
  await server.close();
  await rm(temporary,{recursive:true,force:true});
}
