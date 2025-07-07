import simpleaudio as sa
import os

wave_obj = sa.WaveObject.from_wave_file(os.path.join(os.path.dirname(__file__), "alarm.wav"))
play_obj = wave_obj.play()
play_obj.wait_done()  # počakaj da konča