window.SD = (() => {
  /*
   * Painterro is made a field of the SD global object
   * To provide convinience when using w() method in css_and_js.py
   */
  class PainterroClass {
    static isOpen = false;
    static async init (toId) {
      const img = SD.x;
      const originalImage = Array.isArray(img) ? img[0] : img;

      if (window.Painterro === undefined) {
        try {
          await this.load();
        } catch (e) {
          SDClass.error(e);

          return this.fallback(originalImage);
        }
      }

      if (this.isOpen) {
        return this.fallback(originalImage);
      }
      this.isOpen = true;

      let resolveResult;
      const paintClient = Painterro({
        hiddenTools: ['arrow'],
        onHide: () => {
          resolveResult?.(null);
        },
        saveHandler: (image, done) => {
          const data = image.asDataURL();

          // ensures stable performance even
          // when the editor is in interactive mode
          SD.clearImageInput(SD.el.get(`#${toId}`));

          resolveResult(data);

          done(true);
          paintClient.hide();
        },
      });

      const result = await new Promise((resolve) => {
        resolveResult = resolve;
        paintClient.show(originalImage);
      });
      this.isOpen = false;

      return result ? this.success(result) : this.fallback(originalImage);
    }
    static success (result) { return [result, result]; }
    static fallback (image) { return [image, image]; }
    static load () {
      return new Promise((resolve, reject) => {
        const scriptId = '__painterro-script';
        if (document.getElementById(scriptId)) {
          reject(new Error('Tried to load painterro script, but script tag already exists.'));
          return;
        }

        const styleId = '__painterro-css-override';
        if (!document.getElementById(styleId)) {
          /* Ensure Painterro window is always on top */
          const style = document.createElement('style');
          style.id = styleId;
          style.setAttribute('type', 'text/css');
          style.appendChild(document.createTextNode(`
            .ptro-holder-wrapper {
                z-index: 100;
            }
          `));
          document.head.appendChild(style);
        }

        const script = document.createElement('script');
        script.id = scriptId;
        script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
        script.onload = () => resolve(true);
        script.onerror = (e) => {
          // remove self on error to enable reattempting load
          document.head.removeChild(script);
          reject(e);
        };
        document.head.appendChild(script);
      });
    }
  }

  /*
   * Turns out caching elements doesn't actually work in gradio
   * As elements in tabs might get recreated
   */
  class ElementCache {
    #el;
    constructor () {
      this.root = document.querySelector('gradio-app').shadowRoot;
    }
    get (selector) {
      return this.root.querySelector(selector);
    }
  }

  /*
   * The main helper class to incapsulate functions
   * that change gradio ui functionality
   */
  class SDClass {
    el = new ElementCache();
    x;
    Painterro = PainterroClass;
    with (x) {
      this.x = x;
      return this;
    }
    moveImageFromGallery (fromId, toId) {
      if (!Array.isArray(this.x) || this.x.length === 0) return;

      this.clearImageInput(this.el.get(`#${toId}`));

      const i = this.#getGallerySelectedIndex(this.el.get(`#${fromId}`));

      return [this.x[i].replace('data:;','data:image/png;')];
    }
    async copyImageFromGalleryToClipboard (fromId) {
      if (!Array.isArray(this.x) || this.x.length === 0) return;

      const i = this.#getGallerySelectedIndex(this.el.get(`#${fromId}`));

      const data = this.x[i];
      const blob = await (await fetch(data.replace('data:;','data:image/png;'))).blob();
      const item = new ClipboardItem({'image/png': blob});

      try {
        navigator.clipboard.write([item]);
      } catch (e) {
        SDClass.error(e);
      }

      return this.x;
    }
    clearImageInput (imageEditor) {
      imageEditor?.querySelector('.modify-upload button:last-child')?.click();
    }
    clickFirstVisibleButton(rowId) {
      const generateButtons = this.el.get(`#${rowId}`).querySelectorAll('.gr-button-primary');

      if (!generateButtons) return;

      for (let i = 0, arr = [...generateButtons]; i < arr.length; i++) {
        const cs = window.getComputedStyle(arr[i]);

        if (cs.display !== 'none' && cs.visibility !== 'hidden') {
          console.log(arr[i]);

          arr[i].click();
          break;
        }
      }
    }
    static error (e) {
      console.error(e);
      if (typeof e === 'string') {
        alert(e);
      } else if(typeof e === 'object' && Object.hasOwn(e, 'message')) {
        alert(e.message);
      }
    }
    #getGallerySelectedIndex (gallery) {
      const selected = gallery.querySelector(`.\\!ring-2`);
      return selected ? [...selected.parentNode.children].indexOf(selected) : 0;
    }
  }

  return new SDClass();
})();
