import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

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
        batch = int(post_data['batch'])
        steps = int(post_data['steps'])
        width = int(post_data['width'])
        height = int(post_data['height'])
        cfgscale = float(post_data['cfgscale'])
        seed = None if int(post_data['seed']) == -1 else int(post_data['seed'])

        print(f"Request to generate with data: {post_data}")
        outputs = model.txt2img(prompt,
                                batch_size = batch,
                                cfg_scale = cfgscale,
                                width = width,
                                height = height,
                                seed = seed,
                                steps = steps);
        print(f"Prompt generated with output: {outputs}")

        outputs = [x + [prompt] for x in outputs] # Append prompt to each output
        result = {'outputs': outputs}
        self.wfile.write(bytes(json.dumps(result), "utf-8"))

if __name__ == "__main__":
    dream_server = HTTPServer(("0.0.0.0", 9090), DreamServer)
    print("Started Stable Diffusion dream server!")

    try:
        dream_server.serve_forever()
    except KeyboardInterrupt:
        pass

    dream_server.server_close()

