# polybench-javascript: the exercism JS track's devDependencies are uniform
# across all exercises (jest + babel + core-js). Install them once into
# /node_modules; Node's upward module resolution lets every exercise mounted
# at /work find them with zero per-run npm install.
FROM node:20
WORKDIR /opt/jsdeps
RUN cat > package.json <<'EOF'
{
  "name": "polybench-jsdeps",
  "private": true,
  "devDependencies": {
    "@babel/core": "^7.25.2",
    "@exercism/babel-preset-javascript": "^0.2.1",
    "@exercism/eslint-config-javascript": "^0.6.0",
    "@types/jest": "^29.5.12",
    "@types/node": "^20.12.12",
    "babel-jest": "^29.6.4",
    "core-js": "~3.37.1",
    "eslint": "^8.49.0",
    "jest": "^29.7.0"
  }
}
EOF
RUN npm install --no-audit --no-fund --loglevel=error \
 && mv node_modules /node_modules \
 && cd / && rm -rf /opt/jsdeps
ENV NODE_PATH=/node_modules
