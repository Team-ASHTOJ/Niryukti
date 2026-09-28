#!/usr/bin/env node
'use strict';
const {spawn} = require('node:child_process');
const {executable} = require('./index');
if (process.argv[2] === 'report') {
  const fs=require('node:fs');const args=process.argv.slice(3);
  const index=args.indexOf('--output');
  if (!args[0] || index<0 || !args[index+1]) {
    console.error('Usage: niryukti report result.json --output report.html');process.exit(1);
  }
  try {
    const {renderReport}=require('./report');
    fs.writeFileSync(args[index+1],renderReport(JSON.parse(fs.readFileSync(args[0],'utf8'))));
    console.log(args[index+1]);process.exit(0);
  } catch(error) {console.error(error.message);process.exit(1);}
}
const child = spawn(executable(), process.argv.slice(2), {stdio:'inherit'});
for (const signal of ['SIGINT','SIGTERM']) process.on(signal, () => child.kill(signal));
child.on('error', error => {console.error(error.message); process.exitCode=1;});
child.on('close', code => {process.exitCode = code ?? 1;});
