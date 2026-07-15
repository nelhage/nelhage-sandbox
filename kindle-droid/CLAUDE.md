# CLAUDE.md — kindle-droid

## frida gotchas
- **frida-python lowercases/snake_cases RPC export names.** A JS agent export
  `rpc.exports = { removeDownload() {} }` is called from Python as
  `script.exports_sync.remove_download(...)` — NOT `.removeDownload(...)`, which
  fails with `RPCException: unable to find method 'removedownload'`. So in
  `harvest.py`, `agent_call(dev, "remove_download", ...)` (snake_case), even
  though the JS name is `removeDownload`. Single-word exports (`download`,
  `open`, `curasin`) are unaffected.
- Rebuilding the agent: `krx_agent.js` is `frida-compile`d from `agent_src.js`
  (ESM `import Java from 'frida-java-bridge'`). Build from `scratch/` (has the
  npm deps): `cp agent_src.js scratch/ && cd scratch && ./node_modules/.bin/frida-compile agent_src.js -o ../krx_agent.js`.
  frida-compile requires the entrypoint to live inside the project root, hence
  the copy into `scratch/`.
- A method that reflects fine (shows up in `getDeclaredMethods()`) can still be
  un-callable on the wrapper frida binds from an interface getter — cast to the
  concrete class first, e.g. `Java.cast(sdk.getLibraryManager(), Java.use('com.amazon.kindle.krx.library.LibraryManager'))`.
