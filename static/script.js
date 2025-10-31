// 🌿 Simple JS Validation for LeafScan

// Upload form
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');

if (uploadForm) {
  uploadForm.onsubmit = function () {
    if (!fileInput.value) {
      alert('Please select a file before submitting!');
      return false;
    }
  };
}

// Feedback form
const feedbackForm = document.getElementById('feedbackForm');
if (feedbackForm) {
  feedbackForm.onsubmit = function () {
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const message = document.getElementById('message').value;

    if (!name || !email || !message) {
      alert('Please fill in all fields before submitting!');
      return false;
    } else {
      alert('Thank you! Your message has been submitted.');
    }
  };
}
