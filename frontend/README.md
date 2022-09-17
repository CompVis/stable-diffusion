# Stable Diffusion Web UI

## Build

from `frontend/`:

- `yarn dev` runs vite dev server
- `yarn build-dev` builds dev
- `yarn build` builds prod

from `.`:

- `python backend/server.py` serves both frontend and backend at http://localhost:9090

## TODO

- Search repo for "TODO"
- My one gripe with Chakra: no way to disable all animations right now and drop the dependence on
  `framer-motion`. I would prefer to save the ~30kb on bundle and have zero animations. This is on
  the Chakra roadmap. See https://github.com/chakra-ui/chakra-ui/pull/6368 for last discussion on
  this. Need to check in on this issue periodically.
- More status info e.g. phase of processing, image we are on of the total count, etc
- Mobile friendly layout
- Proper image gallery/viewer/manager
- Instead of deleting images directly, use something like [send2trash](https://pypi.org/project/Send2Trash/)
