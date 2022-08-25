import json
import base64
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

print("Loading model...")
from ldm.simplet2i import T2I
model = T2I()

class DreamServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        if self.path == "/":
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("./scripts/static/index.html", "rb") as content:
                self.wfile.write(content.read())
        else:
            self.send_header("Content-type", "image/png")
            self.end_headers()
            with open("." + self.path, "rb") as content:
                self.wfile.write(content.read())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        prompt = post_data['prompt']
        initimg = post_data['initimg']
        batch = int(post_data['batch'])
        steps = int(post_data['steps'])
        width = int(post_data['width'])
        height = int(post_data['height'])
        cfgscale = float(post_data['cfgscale'])
        seed = None if int(post_data['seed']) == -1 else int(post_data['seed'])

        print(f"Request to generate with prompt: {prompt}")

        outputs = []
        if initimg is None:
            # Run txt2img
            outputs = model.txt2img(prompt,
                                    batch_size = batch,
                                    cfg_scale = cfgscale,
                                    width = width,
                                    height = height,
                                    seed = seed,
                                    steps = steps)
        else:
            # Decode initimg as base64 to temp file
            with open("./img2img-tmp.png", "wb") as f:
                initimg = initimg.split(",")[1] # Ignore mime type
                f.write(base64.b64decode(initimg))

                # Run img2img
                outputs = model.img2img(prompt,
                                        init_img = "./img2img-tmp.png",
                                        batch_size = batch,
                                        cfg_scale = cfgscale,
                                        seed = seed,
                                        steps = steps)
            # Remove the temp file
            os.remove("./img2img-tmp.png")

        print(f"Prompt generated with output: {outputs}")

        post_data['initimg'] = '' # Don't send init image back
        outputs = [x + [post_data] for x in outputs] # Append config to each output
        result = {'outputs': outputs}
        self.wfile.write(bytes(json.dumps(result), "utf-8"))

if __name__ == "__main__":
    dream_server = ThreadingHTTPServer(("0.0.0.0", 9090), DreamServer)
    print("Started Stable Diffusion dream server!")

    try:
        dream_server.serve_forever()
    except KeyboardInterrupt:
        pass

    dream_server.server_close()

