import http from 'node:http';
import {readFile, realpath} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const types = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.mp4':'video/mp4','.webm':'video/webm','.md':'text/plain; charset=utf-8'};
export function startServer(port = 0) {
return new Promise((resolve, reject) => {
const server = http.createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://local').pathname);
    const requested = path.resolve(root, '.' + (pathname.endsWith('/') ? pathname+'index.html' : pathname));
    const resolved = await realpath(requested);
    if (!resolved.startsWith(root + path.sep)) {res.writeHead(403); res.end(); return;}
    const data = await readFile(resolved);
    res.writeHead(200, {'Content-Type':types[path.extname(resolved)] || 'application/octet-stream','Cache-Control':'no-store'});
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch {
    res.writeHead(404, {'Content-Type':'text/plain; charset=utf-8'}); res.end('Not found');
  }
});
server.once('error', reject);
server.listen(port, '127.0.0.1', () => resolve({
  url: `http://127.0.0.1:${server.address().port}`,
  close: () => new Promise((done, fail) => server.close(error => error ? fail(error) : done())),
}));
});
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const server = await startServer(Number(process.env.PORT || 4173));
  console.log(`S-PROJECT: ${server.url}`);
}
