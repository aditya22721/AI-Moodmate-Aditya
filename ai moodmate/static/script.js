const form = document.getElementById('moodForm');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = document.getElementById('userText').value;

  const formData = new FormData();
  formData.append('text', text);

  const response = await fetch('/get_recommendations', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  document.getElementById('emotion').textContent = data.emotion;

  const songsList = document.getElementById('songs');
  songsList.innerHTML = '';
  data.songs.forEach(song => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${song.track}</strong> by ${song.artist} [${song.seeds}] - <a href="${song.lastfm_url}" target="_blank">Listen</a>`;
    songsList.appendChild(li);
  });
});

// Poll webcam emotion
setInterval(async () => {
  const res = await fetch('/emotion');
  const data = await res.json();
  document.getElementById('emotion').textContent = data.emotion;
}, 1000);