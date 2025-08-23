// in_out_script.js

const video = document.getElementById('video');
const statusDiv = document.getElementById('status');
const overlayCanvas = document.getElementById('overlayCanvas');
const context = overlayCanvas.getContext('2d');
const captureCanvas = document.createElement('canvas');
const markButton = document.getElementById('markButton');

const endpoint = markButton.dataset.endpoint;
if (endpoint.includes('out')) {
    statusDiv.textContent = 'Ready to check OUT.';
} else {
    statusDiv.textContent = 'Ready to check IN.';
}

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
        video.onloadeddata = () => {
            overlayCanvas.width = video.videoWidth;
            overlayCanvas.height = video.videoHeight;
        };
    })
    .catch(err => {
        console.error("Error accessing the webcam: " + err);
        statusDiv.textContent = "Error: Could not access webcam.";
    });

markButton.addEventListener('click', () => {
    statusDiv.textContent = "Processing...";
    markButton.disabled = true;

    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        statusDiv.textContent = "Video not ready.";
        markButton.disabled = false;
        return;
    }

    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureContext = captureCanvas.getContext('2d');
    captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    
    captureCanvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'face.jpg');

        fetch(`http://127.0.0.1:5000${endpoint}`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.message || 'Network response was not ok');
                });
            }
            return response.json();
        })
        .then(data => {
            context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            if (data.box_coords) {
                const [x1, y1, x2, y2] = data.box_coords;
                const scaleX = overlayCanvas.width / video.videoWidth;
                const scaleY = overlayCanvas.height / video.videoHeight;

                context.strokeStyle = data.box_color;
                context.lineWidth = 2;
                context.beginPath();
                context.rect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
                context.stroke();

                context.fillStyle = data.box_color;
                context.font = '16px Arial';
                context.fillText(data.recognized_id, x1 * scaleX, (y1 > 10 ? y1 - 5 : 10) * scaleY);
            }
            statusDiv.textContent = data.message;
            if (data.recognized_id !== "N/A") {
                if (data.box_color === 'green') {
                    statusDiv.style.color = 'green';
                } else {
                    statusDiv.style.color = 'red';
                }
            }
            markButton.disabled = false;
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
            statusDiv.textContent = `Request failed: ${error.message}`;
            statusDiv.style.color = 'red';
            markButton.disabled = false;
        });
    }, 'image/jpeg');
});