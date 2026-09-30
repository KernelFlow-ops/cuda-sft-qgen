process.env.SPEC_DIR = 'slides';
process.env.OUT_DIR = process.env.OUT_DIR || require('path').join(__dirname, '..', 'slides');
require('./build.js');
