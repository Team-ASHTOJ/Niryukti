'use strict';
const {spawn} = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
function executable(binary) {
  return binary || process.env.VANTAGE_BINARY || path.join(__dirname, 'native', 'vantage');
}
async function solve(model, options = {}) {
  const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'vantage-node-'));
  try {
    let input = model;
    if (typeof model !== 'string') {
      input = path.join(folder, 'model.json');
      await fs.writeFile(input, JSON.stringify(model));
    }
    const output = path.join(folder, 'result.json');
    const args = ['solve', input, '--json-out', output, '--device', options.device || 'cpu',
      '--method', options.method || 'auto', '--time-limit', String(options.timeLimit ?? 60),
      '--tol', String(options.tolerance ?? 1e-6), '--threads', String(options.threads ?? 1)];
    await new Promise((resolve,reject) => {
      const child = spawn(executable(options.binary), args, {signal: options.signal});
      let errors = '';
      child.stdout.resume();
      child.stderr.on('data', chunk => { errors = (errors + chunk).slice(-8192); });
      child.on('error', reject);
      child.on('close', (code, signal) => {
        if (code === 0 || code === 2) resolve();
        else reject(new Error(`VANTAGE failed (${signal || code}): ${errors}`));
      });
    });
    return JSON.parse(await fs.readFile(output, 'utf8'));
  } finally { await fs.rm(folder, {recursive:true, force:true}); }
}
module.exports = {solve, executable};
