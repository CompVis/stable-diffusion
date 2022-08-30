from os import path

def readTextFile(*args):
    dir = path.dirname(__file__)
    entry = path.join(dir, *args)
    with open(entry, "r", encoding="utf8") as f:
        data = f.read()
    return data

def css(opt):
    styling = readTextFile("css", "styles.css")
    if not opt.no_progressbar_hiding:
        styling += readTextFile("css", "no_progress_bar.css")
    return styling

def js(opt):
    data = readTextFile("js", "index.js")
    data = "(z) => {" + data + "; return z ?? [] }"
    return data

# Wrap the typical SD method call into async closure for ease of use
# If you call frontend method without wrapping
# DONT FORGET to bind input argument if you need it: SD.with(x)
def w(sd_method_call):
    return f"async (x) => {{ return await SD.with(x).{sd_method_call} ?? x ?? []; }}"

def js_move_image(from_id, to_id):
    return w(f"moveImageFromGallery('{from_id}', '{to_id}')")

def js_copy_to_clipboard(from_id):
    return w(f"copyImageFromGalleryToClipboard('{from_id}')")

def js_painterro_launch(to_id):
    return w(f"Painterro.init('{to_id}')")

def js_img2img_submit(prompt_row_id):
    return w(f"clickFirstVisibleButton('{prompt_row_id}')")
