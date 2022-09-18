# Stable Diffusion Web UI

## Run

- `python backend/server.py` serves both frontend and backend at http://localhost:9090

## Evironment

Install [node](https://nodejs.org/en/download/) (includes npm) and optionally
[yarn](https://yarnpkg.com/getting-started/install).

From `frontend/` run `npm install` / `yarn install` to install the frontend packages.

## Dev

1. From `frontend/`, run `npm dev` / `yarn dev` to start the dev server.
2. Note the address it starts up on (probably `http://localhost:5173/`).
3. Edit `backend/server.py`'s `additional_allowed_origins` to include this address, e.g.
   `additional_allowed_origins = ['http://localhost:5173']`.
4. Leaving the dev server running, open a new terminal and go to the project root.
5. Run `python backend/server.py`.
6. Navigate to the dev server address e.g. `http://localhost:5173/`.

To build for dev: `npm build-dev` / `yarn build-dev`

To build for production: `npm build` / `yarn build`

## TODO

- Search repo for "TODO"
- My one gripe with Chakra: no way to disable all animations right now and drop the dependence on
  `framer-motion`. I would prefer to save the ~30kb on bundle and have zero animations. This is on
  the Chakra roadmap. See https://github.com/chakra-ui/chakra-ui/pull/6368 for last discussion on
  this. Need to check in on this issue periodically.
- Mobile friendly layout
- Proper image gallery/viewer/manager
- Help tooltips and such
