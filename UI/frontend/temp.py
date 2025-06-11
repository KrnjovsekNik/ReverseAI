from pydub import AudioSegment

# Naloži originalni wav
sound = AudioSegment.from_wav("2000.wav")

# Skrajšaj na 1 sekundo (1000 ms)
short_sound = sound[:200]

# Shrani v novo datoteko (ali prepiši isto)
short_sound.export("alarm.wav", format="wav")

print("✅ Zvok skrajšan na 1 sekundo.")