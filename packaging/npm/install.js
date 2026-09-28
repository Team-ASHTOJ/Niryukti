'use strict';
const {spawnSync} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const build = path.join(__dirname, 'native');
if (process.env.VANTAGE_BINARY || fs.existsSync(path.join(build, 'vantage'))) process.exit(0);
for (const args of [
  ['-S', path.join(__dirname,'engine'), '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF', '-DVANTAGE_OPENMP=OFF'],
  ['--build', build, '--target', 'vantage_cli', '--parallel', '2']
]) {
  const result = spawnSync('cmake', args, {stdio:'inherit'});
  if (result.error || result.status !== 0) {
    console.error('VANTAGE source install needs CMake >=3.24 and a C++20 compiler. Alternatively set VANTAGE_BINARY.');
    process.exit(1);
  }
}
