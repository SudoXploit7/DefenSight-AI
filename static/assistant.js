function toggleChat() {
  const box = document.getElementById("chat-box");
  box.style.display = (box.style.display === "none") ? "block" : "none";
  if (box.style.display === "block") {
    document.getElementById("user-input").focus();
  }
}

function sendChat() {
  const input = document.getElementById("user-input");
  const msg = input.value.trim();
  if (!msg) return;

  const chat = document.getElementById("chat-messages");
  chat.innerHTML += `<div class="mb-2"><strong>You:</strong> ${escapeHtml(msg)}</div>`;
  chat.innerHTML += `<div id="typing" class="text-muted"><em>Assistant is typing<span class="loading-dots">...</span></em></div>`;
  input.value = "";
  input.disabled = true;
  chat.scrollTop = chat.scrollHeight;

  fetch("/chat", {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: msg })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("typing").remove();
    console.log("🧠 Received reply:", data);

    const replyHtml = data.reply || "<em>No reply</em>";  // <-- ✅ use directly
    chat.innerHTML += `<div class="mb-3"><strong>GPT:</strong><div class="mt-1">${replyHtml}</div></div>`;

    input.disabled = false;
    input.focus();
    chat.scrollTop = chat.scrollHeight;
  })
  .catch(() => {
    const typing = document.getElementById("typing");
    if (typing) typing.innerHTML = "<span class='text-danger'>⚠️ Failed to get response. Please try again.</span>";
    input.disabled = false;
  });
}


// Escape user input for safe HTML rendering
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, function (m) {
    return ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    })[m];
  });
}
