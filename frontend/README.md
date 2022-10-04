# Stable Diffusion Web UI

## Run

- `python scripts/dream.py --web` serves both frontend and backend at
  http://localhost:9090

## Evironment

Install [node](https://nodejs.org/en/download/) (includes npm) and optionally
[yarn](https://yarnpkg.com/getting-started/install).

From `frontend/` run `npm install` / `yarn install` to install the frontend
packages.

## Dev

1. From `frontend/`, run `npm dev` / `yarn dev` to start the dev server.
2. Run `python scripts/dream.py --web`.
3. Navigate to the dev server address e.g. `http://localhost:5173/`.

To build for dev: `npm build-dev` / `yarn build-dev`

To build for production: `npm build` / `yarn build`

## TODO

- Search repo for "TODO"
