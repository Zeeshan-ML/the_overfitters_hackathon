document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const fileInput = document.getElementById('file');
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                // Create a temporary URL for the image preview
                const imageURL = URL.createObjectURL(file);

                // Build a rich HTML result card
                const resultHTML = `
                  <div class="test-result">
                      <img src="${imageURL}" alt="Uploaded Image">
                      <div class="result-info">
                          <h2>Prediction Result</h2>
                          <p><strong>Filename:</strong> ${result.filename}</p>
                          <p><strong>Predicted Label:</strong> ${result.prediction}</p>
                          <p><strong>Raw Predictions:</strong></p>
                          <pre>${JSON.stringify(result.raw_prediction, null, 2)}</pre>
                      </div>
                  </div>
                `;
                document.getElementById('result').innerHTML = resultHTML;
            } catch (error) {
                document.getElementById('result').innerText = "Error: " + error;
            }
        });
    }
});
