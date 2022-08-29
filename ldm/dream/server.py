import json
import base64
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class DreamServer(BaseHTTPRequestHandler):
    model = None

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("./static/dream_web/index.html", "rb") as content:
                self.wfile.write(content.read())
        else:
            path = "." + self.path
            cwd = os.getcwd()
            is_in_cwd = os.path.commonprefix((os.path.realpath(path), cwd)) == cwd
            if not (is_in_cwd and os.path.exists(path)):
                self.send_response(404)
                return
            mime_type = mimetypes.guess_type(path)[0]
            if mime_type is not None:
                self.send_response(200)
                self.send_header("Content-type", mime_type)
                self.end_headers()
                with open("." + self.path, "rb") as content:
                    self.wfile.write(content.read())
            else:
                self.send_response(404)

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        prompt = post_data['prompt']
        initimg = post_data['initimg']
        iterations = int(post_data['iterations'])
        steps = int(post_data['steps'])
        width = int(post_data['width'])
        height = int(post_data['height'])
        cfgscale = float(post_data['cfgscale'])
        gfpgan_strength = float(post_data['gfpgan_strength'])
        upscale_level    = post_data['upscale_level']
        upscale_strength = post_data['upscale_strength']
        upscale = [int(upscale_level),float(upscale_strength)] if upscale_level != '' else None
        seed = None if int(post_data['seed']) == -1 else int(post_data['seed'])

        print(f"Request to generate with prompt: {prompt}")

        outputs = []
        if initimg is None:
            # Run txt2img
            outputs = self.model.txt2img(prompt,
                                    iterations=iterations,
                                    cfg_scale = cfgscale,
                                    width = width,
                                    height = height,
                                    seed = seed,
                                    steps = steps,
                                    gfpgan_strength = gfpgan_strength,
                                    upscale         = upscale
            )
        else:
            # Decode initimg as base64 to temp file
            with open("./img2img-tmp.png", "wb") as f:
                initimg = initimg.split(",")[1] # Ignore mime type
                f.write(base64.b64decode(initimg))

            # Run img2img
            outputs = self.model.img2img(prompt,
                                         init_img = "./img2img-tmp.png",
                                         iterations = iterations,
                                         cfg_scale = cfgscale,
                                         seed = seed,
                                         gfpgan_strength=gfpgan_strength,
                                         upscale         = upscale,
                                         steps = steps
            )
            # Remove the temp file
            os.remove("./img2img-tmp.png")

        print(f"Prompt generated with output: {outputs}")

        post_data['initimg'] = '' # Don't send init image back

        # Append post_data to log
        with open("./outputs/img-samples/dream_web_log.txt", "a", encoding="utf-8") as log:
            for output in outputs:
                log.write(f"{output[0]}: {json.dumps(post_data)}\n")

        outputs = [x + [post_data] for x in outputs] # Append config to each output
        result = {'outputs': outputs}
        self.wfile.write(bytes(json.dumps(result), "utf-8"))


class ThreadingDreamServer(ThreadingHTTPServer):
    def __init__(self, server_address):
        super(ThreadingDreamServer, self).__init__(server_address, DreamServer)
