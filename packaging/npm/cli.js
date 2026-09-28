#!/usr/bin/env node
'use strict';
const {spawn} = require('node:child_process');
const {executable} = require('./index');
const child = spawn(executable(), process.argv.slice(2), {stdio:'inherit'});
for (const signal of ['SIGINT','SIGTERM']) process.on(signal, () => child.kill(signal));
child.on('error', error => {console.error(error.message); process.exitCode=1;});
child.on('close', code => {process.exitCode = code ?? 1;});
