# Stable Diffusion Web UI

Demo at https://peaceful-otter-7a427f.netlify.app/ (not connected to back end)

much of this readme is just notes for myself during dev work

numpy rand: 0 to 4294967295

## Test and Build

from `frontend/`:

-   `yarn dev` runs `tsc-watch`, which runs `vite build` on successful `tsc` transpilation

from `.`:

-   `python backend/server.py` serves both frontend and backend at http://localhost:9090

## API

`backend/server.py` serves the UI and provides a [socket.io](https://github.com/socketio/socket.io) API via [flask-socketio](https://github.com/miguelgrinberg/flask-socketio).

### Server Listeners

The server listens for these socket.io events:

`cancel`

-   Cancels in-progress image generation
-   Returns ack only

`generateImage`

-   Accepts object of image parameters
-   Generates an image
-   Returns ack only (image generation function sends progress and result via separate events)

`deleteImage`

-   Accepts file path to image
-   Deletes image
-   Returns ack only

`deleteAllImages` WIP

-   Deletes all images in `outputs/`
-   Returns ack only

`requestAllImages`

-   Returns array of all images in `outputs/`

`requestCapabilities` WIP

-   Returns capabilities of server (torch device, GFPGAN and ESRGAN availability, ???)

`sendImage` WIP

-   Accepts a File and attributes
-   Saves image
-   Used to save init images which are not generated images

### Server Emitters

`progress`

-   Emitted during each step in generation
-   Sends a number from 0 to 1 representing percentage of steps completed

`result` WIP

-   Emitted when an image generation has completed
-   Sends a object:

```
{
    url: relative_file_path,
    metadata: image_metadata_object
}
```

## TODO

-   Search repo for "TODO"
-   My one gripe with Chakra: no way to disable all animations right now and drop the dependence on `framer-motion`. I would prefer to save the ~30kb on bundle and have zero animations. This is on the Chakra roadmap. See https://github.com/chakra-ui/chakra-ui/pull/6368 for last discussion on this. Need to check in on this issue periodically.
