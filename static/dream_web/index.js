function toBase64(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.readAsDataURL(file);
        r.onload = () => resolve(r.result);
        r.onerror = (error) => reject(error);
    });
}

function appendOutput(output) {
    let outputNode = document.createElement("img");
    outputNode.src = output[0];

    let outputConfig = output[2];
    let altText = output[1].toString() + " | " + outputConfig.prompt;
    outputNode.alt = altText;
    outputNode.title = altText;

    // Reload image config
    outputNode.addEventListener('click', () => {
        let form = document.querySelector("#generate-form");
        for (const [k, v] of new FormData(form)) {
            form.querySelector(`*[name=${k}]`).value = outputConfig[k];
        }
        document.querySelector("#seed").value = output[1];

        saveFields(document.querySelector("#generate-form"));
    });

    document.querySelector("#results").prepend(outputNode);
}

function appendOutputs(outputs) {
    for (const output of outputs) {
        appendOutput(output);
    }
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
    localStorage.clear()
    location.reload()
}

async function generateSubmit(form) {
    const prompt = document.querySelector("#prompt").value;

    // Convert file data to base64
    let formData = Object.fromEntries(new FormData(form));
    formData.initimg = formData.initimg.name !== '' ? await toBase64(formData.initimg) : null;

    // Post as JSON
    fetch(form.action, {
        method: form.method,
        body: JSON.stringify(formData),
    }).then(async (result) => {
        let data = await result.json();

        // Re-enable form, remove no-results-message
        form.querySelector('fieldset').removeAttribute('disabled');
        document.querySelector("#prompt").value = prompt;

        if (data.outputs.length != 0) {
            document.querySelector("#no-results-message")?.remove();
            appendOutputs(data.outputs);
        } else {
            alert("Error occurred while generating.");
        }
    });

    // Disable form while generating
    form.querySelector('fieldset').setAttribute('disabled','');
    document.querySelector("#prompt").value = `Generating: "${prompt}"`;
}

window.onload = () => {
    document.querySelector("#generate-form").addEventListener('submit', (e) => {
        e.preventDefault();
        const form = e.target;

        generateSubmit(form);
    });
    document.querySelector("#generate-form").addEventListener('change', (e) => {
        saveFields(e.target.form);
    });
    document.querySelector("#reset").addEventListener('click', (e) => {
        document.querySelector("#seed").value = -1;
        saveFields(e.target.form);
    });
    document.querySelector("#reset-all").addEventListener('click', (e) => {
        clearFields(e.target.form);
    });
    loadFields(document.querySelector("#generate-form"));
};
