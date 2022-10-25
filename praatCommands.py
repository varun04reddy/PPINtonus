import ipywidgets
import glob
import parselmouth
from parselmouth.praat import call

from IPython.display import Audio

sound = parselmouth.Sound("audio/4_b.wav")

Audio(data=sound.values, rate=sound.sampling_frequency)

def interactive_change_pitch(audio_file, factor):
    sound = parselmouth.Sound(audio_file)
    sound_changed_pitch = change_pitch(sound, factor)
    return Audio(data=sound_changed_pitch.values, rate=sound_changed_pitch.sampling_frequency)

w = ipywidgets.interact(interactive_change_pitch,
                        audio_file=ipywidgets.Dropdown(options=sorted(glob.glob("audio/*.wav")), value="audio/4_b.wav"),
                        factor=ipywidgets.FloatSlider(min=0.25, max=4, step=0.05, value=1.5))

manipulation.class_name

def change_pitch(sound, factor):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    pitch_tier = call(manipulation, "Extract pitch tier")

    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)

    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")


pitch_tier = call(manipulation, "Extract pitch tier")

call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2)

call([pitch_tier, manipulation], "Replace pitch tier")
sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")


manipulation = call(sound, "To Manipulation", 0.01, 75, 600)