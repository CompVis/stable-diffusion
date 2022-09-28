const socket = io();

var priorResultsLoadState = {
  page: 0,
  pages: 1,
  per_page: 10,
  total: 20,
  offset: 0, // number of items generated since last load
  loading: false,
  initialized: false
};

function loadPriorResults() {
  // Fix next page by offset
  let offsetPages = priorResultsLoadState.offset / priorResultsLoadState.per_page;
  priorResultsLoadState.page += offsetPages;
  priorResultsLoadState.pages += offsetPages;
  priorResultsLoadState.total += priorResultsLoadState.offset;
  priorResultsLoadState.offset = 0;

  if (priorResultsLoadState.loading) {
    return;
  }

  if (priorResultsLoadState.page >= priorResultsLoadState.pages) {
    return; // Nothing more to load
  }

  // Load
  priorResultsLoadState.loading = true
  let url = new URL('/api/images', document.baseURI);
  url.searchParams.append('page', priorResultsLoadState.initialized ? priorResultsLoadState.page + 1 : priorResultsLoadState.page);
  url.searchParams.append('per_page', priorResultsLoadState.per_page);
  fetch(url.href, {
    method: 'GET',
    headers: new Headers({'content-type': 'application/json'})
  })
  .then(response => response.json())
  .then(data => {
    priorResultsLoadState.page = data.page;
    priorResultsLoadState.pages = data.pages;
    priorResultsLoadState.per_page = data.per_page;
    priorResultsLoadState.total = data.total;

    data.items.forEach(function(dreamId, index) {
      let src = 'api/images/' + dreamId;
      fetch('/api/images/' + dreamId + '/metadata', {
        method: 'GET',
        headers: new Headers({'content-type': 'application/json'})
      })
      .then(response => response.json())
      .then(metadata => {
        let seed = metadata.seed || 0; // TODO: Parse old metadata
        appendOutput(src, seed, metadata, true);
      });
    });
    
    // Load until page is full
    if (!priorResultsLoadState.initialized) {
      if (document.body.scrollHeight <= window.innerHeight) {
        loadPriorResults();
      }
    }
  })
  .finally(() => {
    priorResultsLoadState.loading = false;
    priorResultsLoadState.initialized = true;
  });
}

function resetForm() {
  var form = document.getElementById('generate-form');
  form.querySelector('fieldset').removeAttribute('disabled');
}

function initProgress(totalSteps, showProgressImages) {
  // TODO: Progress could theoretically come from multiple jobs at the same time (in the future)
  let progressSectionEle = document.querySelector('#progress-section');
  progressSectionEle.style.display = 'initial';
  let progressEle = document.querySelector('#progress-bar');
  progressEle.setAttribute('max', totalSteps);

  let progressImageEle = document.querySelector('#progress-image');
  progressImageEle.src = BLANK_IMAGE_URL;
  progressImageEle.style.display = showProgressImages ? 'initial': 'none';
}

function setProgress(step, totalSteps, src) {
  let progressEle = document.querySelector('#progress-bar');
  progressEle.setAttribute('value', step);

  if (src) {
    let progressImageEle = document.querySelector('#progress-image');
    progressImageEle.src = src;
  }
}

function resetProgress(hide = true) {
  if (hide) {
    let progressSectionEle = document.querySelector('#progress-section');
    progressSectionEle.style.display = 'none';
  }
  let progressEle = document.querySelector('#progress-bar');
  progressEle.setAttribute('value', 0);
}

function toBase64(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.readAsDataURL(file);
        r.onload = () => resolve(r.result);
        r.onerror = (error) => reject(error);
    });
}

function ondragdream(event) {
  let dream = event.target.dataset.dream;
  event.dataTransfer.setData("dream", dream);
}

function seedClick(event) {
    // Get element
    var image = event.target.closest('figure').querySelector('img');
    var dream = JSON.parse(decodeURIComponent(image.dataset.dream));

    let form = document.querySelector("#generate-form");
    for (const [k, v] of new FormData(form)) {
        if (k == 'initimg') { continue; }
        let formElem = form.querySelector(`*[name=${k}]`);
        formElem.value = dream[k] !== undefined ? dream[k] : formElem.defaultValue;
    }

    document.querySelector("#seed").value = dream.seed;
    document.querySelector('#iterations').value = 1; // Reset to 1 iteration since we clicked a single image (not a full job)

    // NOTE: leaving this manual for the user for now - it was very confusing with this behavior
    // document.querySelector("#with_variations").value = variations || '';
    // if (document.querySelector("#variation_amount").value <= 0) {
    //     document.querySelector("#variation_amount").value = 0.2;
    // }

    saveFields(document.querySelector("#generate-form"));
}

function appendOutput(src, seed, config, toEnd=false) {
    let outputNode = document.createElement("figure");
    let altText = seed.toString() + " | " + config.prompt;

    // img needs width and height for lazy loading to work
    // TODO: store the full config in a data attribute on the image?
    const figureContents = `
        <a href="${src}" target="_blank">
            <img src="${src}"
                 alt="${altText}"
                 title="${altText}"
                 loading="lazy"
                 width="256"
                 height="256"
                 draggable="true"
                 ondragstart="ondragdream(event, this)"
                 data-dream="${encodeURIComponent(JSON.stringify(config))}"
                 data-dreamId="${encodeURIComponent(config.dreamId)}">
        </a>
        <figcaption onclick="seedClick(event, this)">${seed}</figcaption>
    `;

    outputNode.innerHTML = figureContents;

    if (toEnd) {
      document.querySelector("#results").append(outputNode);
    } else {
      document.querySelector("#results").prepend(outputNode);
    }
    document.querySelector("#no-results-message")?.remove();
}

function saveFields(form) {
    for (const [k, v] of new FormData(form)) {
        if (typeof v !== 'object') { // Don't save 'file' type
            localStorage.setItem(k, v);
        }
    }
}

function loadFields(form) {
    for (const [k, v] of new FormData(form)) {
        const item = localStorage.getItem(k);
        if (item != null) {
            form.querySelector(`*[name=${k}]`).value = item;
        }
    }
}

function clearFields(form) {
    localStorage.clear();
    let prompt = form.prompt.value;
    form.reset();
    form.prompt.value = prompt;
}

const BLANK_IMAGE_URL = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"/>';
async function generateSubmit(form) {
    // Convert file data to base64
    // TODO: Should probably uplaod files with formdata or something, and store them in the backend?
    let formData = Object.fromEntries(new FormData(form));
    if (!formData.enable_generate && !formData.enable_init_image) {
      gen_label = document.querySelector("label[for=enable_generate]").innerHTML;
      initimg_label = document.querySelector("label[for=enable_init_image]").innerHTML;
      alert(`Error: one of "${gen_label}" or "${initimg_label}" must be set`);
    }


    formData.initimg_name = formData.initimg.name
    formData.initimg = formData.initimg.name !== '' ? await toBase64(formData.initimg) : null;

    // Evaluate all checkboxes
    let checkboxes = form.querySelectorAll('input[type=checkbox]');
    checkboxes.forEach(function (checkbox) {
      if (checkbox.checked) {
        formData[checkbox.name] = 'true';
      }
    });

    let strength = formData.strength;
    let totalSteps = formData.initimg ? Math.floor(strength * formData.steps) : formData.steps;
    let showProgressImages = formData.progress_images;

    // Set enabling flags


    // Initialize the progress bar
    initProgress(totalSteps, showProgressImages);

    // POST, use response to listen for events
    fetch(form.action, {
        method: form.method,
        headers: new Headers({'content-type': 'application/json'}),
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        var jobId = data.jobId;
        socket.emit('join_room', { 'room': jobId });
    });

    form.querySelector('fieldset').setAttribute('disabled','');
}

function fieldSetEnableChecked(event) {
  cb = event.target;
  fields = cb.closest('fieldset');
  fields.disabled = !cb.checked;
}

// Socket listeners
socket.on('job_started', (data) => {})

socket.on('dream_result', (data) => {
  var jobId = data.jobId;
  var dreamId = data.dreamId;
  var dreamRequest = data.dreamRequest;
  var src = 'api/images/' + dreamId;

  priorResultsLoadState.offset += 1;
  appendOutput(src, dreamRequest.seed, dreamRequest);

  resetProgress(false);
})

socket.on('dream_progress', (data) => {
  // TODO: it'd be nice if we could get a seed reported here, but the generator would need to be updated
  var step = data.step;
  var totalSteps = data.totalSteps;
  var jobId = data.jobId;
  var dreamId = data.dreamId;

  var progressType = data.progressType
  if (progressType === 'GENERATION') {
    var src = data.hasProgressImage ?
      'api/intermediates/' + dreamId + '/' + step
      : null;
    setProgress(step, totalSteps, src);
  } else if (progressType === 'UPSCALING_STARTED') {
    // step and totalSteps are used for upscale count on this message
    document.getElementById("processing_cnt").textContent = step;
    document.getElementById("processing_total").textContent = totalSteps;
    document.getElementById("scaling-inprocess-message").style.display = "block";
  } else if (progressType == 'UPSCALING_DONE') {
    document.getElementById("scaling-inprocess-message").style.display = "none";
  }
})

socket.on('job_canceled', (data) => {
  resetForm();
  resetProgress();
})

socket.on('job_done', (data) => {
  jobId = data.jobId
  socket.emit('leave_room', { 'room': jobId });

  resetForm();
  resetProgress();
})

window.onload = async () => {
    document.querySelector("#prompt").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        const form = e.target.form;
        generateSubmit(form);
      }
    });
    document.querySelector("#generate-form").addEventListener('submit', (e) => {
        e.preventDefault();
        const form = e.target;

        generateSubmit(form);
    });
    document.querySelector("#generate-form").addEventListener('change', (e) => {
        saveFields(e.target.form);
    });
    document.querySelector("#reset-seed").addEventListener('click', (e) => {
        document.querySelector("#seed").value = 0;
        saveFields(e.target.form);
    });
    document.querySelector("#reset-all").addEventListener('click', (e) => {
        clearFields(e.target.form);
    });
    document.querySelector("#remove-image").addEventListener('click', (e) => {
        initimg.value=null;
    });
    loadFields(document.querySelector("#generate-form"));

    document.querySelector('#cancel-button').addEventListener('click', () => {
        fetch('/api/cancel').catch(e => {
            console.error(e);
        });
    });
    document.documentElement.addEventListener('keydown', (e) => {
      if (e.key === "Escape")
        fetch('/api/cancel').catch(err => {
          console.error(err);
        });
    });

    if (!config.gfpgan_model_exists) {
        document.querySelector("#gfpgan").style.display = 'none';
    }

    window.addEventListener("scroll", () => {
      if ((window.innerHeight + window.pageYOffset) >= document.body.offsetHeight) {
        loadPriorResults();
      }
    });



    // Enable/disable forms by checkboxes
    document.querySelectorAll("legend > input[type=checkbox]").forEach(function(cb) {
      cb.addEventListener('change', fieldSetEnableChecked);
      fieldSetEnableChecked({ target: cb})
    });


    // Load some of the previous results
    loadPriorResults();

    // Image drop/upload WIP
    /*
    let drop = document.getElementById('dropper');
    function ondrop(event) {
      let dreamData = event.dataTransfer.getData('dream');
      if (dreamData) {
        var dream = JSON.parse(decodeURIComponent(dreamData));
        alert(dream.dreamId);
      }
    };

    function ondragenter(event) {
      event.preventDefault();
    };

    function ondragover(event) {
      event.preventDefault();
    };

    function ondragleave(event) {

    }

    drop.addEventListener('drop', ondrop);
    drop.addEventListener('dragenter', ondragenter);
    drop.addEventListener('dragover', ondragover);
    drop.addEventListener('dragleave', ondragleave);
    */
};
