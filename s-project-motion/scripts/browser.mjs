import {chromium} from 'playwright-core';
import {existsSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
export async function launchBrowser() {
  let executablePath = process.env.CHROME_PATH;
  if (!executablePath) {
    const candidates = ['/usr/bin/chromium','/usr/bin/chromium-browser','/usr/bin/google-chrome',
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      'C:/Program Files/Google/Chrome/Application/chrome.exe',
      'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'];
    executablePath = candidates.find(p => existsSync(p));
  }
  if (!executablePath) {
    for (const name of ['chromium','google-chrome','chrome']) {
      const result = spawnSync(process.platform === 'win32' ? 'where' : 'which', [name], {encoding:'utf8'});
      if (result.status === 0) {executablePath = result.stdout.trim().split(/\r?\n/)[0]; break;}
    }
  }
  if (!executablePath) throw new Error('Chrome/Chromium not found. Install it or set CHROME_PATH to its executable.');
  return chromium.launch({executablePath, headless:true, args:['--no-sandbox','--disable-dev-shm-usage']});
}
