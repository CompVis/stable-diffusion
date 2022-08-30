window.SD = (() => {
  class PainterroClass {
    static init (img) {
      Painterro({
        hiddenTools: ['arrow'],
        saveHandler: function (image, done) {
          localStorage.setItem('painterro-image', image.asDataURL());
          done(true);
        },
      }).show(Array.isArray(img) ? img[0] : img);
    }
    static loadImage (toId) {
      window.SD.clearImageInput(window.SD.el.get(`#${toId}`));

      const image = localStorage.getItem('painterro-image')
      return [image, image];
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
  class SD {
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
        // TODO: graceful error messaing
        console.error(e);
        alert(e.message);
      }

      return this.x;
    }
    clearImageInput (imageEditor) {
      imageEditor?.querySelector('.modify-upload button:last-child')?.click();
    }
    #getGallerySelectedIndex (gallery) {
      const selected = gallery.querySelector(`.\\!ring-2`);
      return selected ? [...selected.parentNode.children].indexOf(selected) : 0;
    }
  }

  // Painterro stuff
  const script = document.createElement('script');
  script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
  document.head.appendChild(script);
  const style = document.createElement('style');
  style.appendChild(document.createTextNode('.ptro-holder-wrapper { z-index: 9999 !important; }'));
  document.head.appendChild(style);

  return new SD();
})();
