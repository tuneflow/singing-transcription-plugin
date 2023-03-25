from __future__ import annotations

from tuneflow_py import TuneflowPlugin, Song, ParamDescriptor, WidgetType, TrackType, InjectSource, Track, Clip, TuneflowPluginTriggerData, ClipAudioDataInjectData
from typing import Any
from data_utils.seq_dataset import SeqDataset
from predictor import EffNetPredictor
import torch
from pathlib import Path
import tempfile
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = EffNetPredictor(device=device, model_path=str(
    Path(__file__).parent.joinpath("models").joinpath("1005_e_4").absolute()))

class TranscribeSinging(TuneflowPlugin):
    @staticmethod
    def provider_id():
        return "hellwz"

    @staticmethod
    def plugin_id():
        return "singing-transcription"

    @staticmethod
    def params(song: Song) -> dict[str, ParamDescriptor]:
        return {
            "clipAudioData": {
                "displayName": {
                    "zh": '音频',
                    "en": 'Audio',
                },
                "defaultValue": None,
                "widget": {
                    "type": WidgetType.NoWidget.value,
                },
                "hidden": True,
                "injectFrom": {
                    "type": InjectSource.ClipAudioData.value,
                    "options": {
                        "clips": "selectedAudioClips"
                    }
                }
            },
            "onsetThreshold": {
                "displayName": {
                    "zh": '音符起始阈值',
                    "en": 'Onset threshold',
                },
                "defaultValue": 0.4,
                "description": {
                    "zh": '该阈值越大，转录出的MIDI音符数越少',
                    "en": 'The higher the threshold, the lower the number of MIDI notes that will be transcribed',
                },
                "widget": {
                    "type": WidgetType.Slider.value,
                    "config": {
                        "minValue": 0.1,
                        "maxValue": 0.9,
                        "step": 0.1
                    }
                },
            },
            "silenceThreshold": {
                "displayName": {
                    "zh": '音符结束阈值',
                    "en": 'Silence threshold',
                },
                "defaultValue": 0.5,
                "description": {
                    "zh": '该阈值越大，转录出的MIDI音符越长',
                    "en": 'The higher the threshold, the longer the MIDI note transcribed',
                },
                "widget": {
                    "type": WidgetType.Slider.value,
                    "config": {
                        "minValue": 0.1,
                        "maxValue": 0.9,
                        "step": 0.1
                    }
                },
            }
        }

    @staticmethod
    def run(song: Song, params: dict[str, Any]):
        trigger: TuneflowPluginTriggerData = params["trigger"]
        trigger_entity_id = trigger["entities"][0]
        track = song.get_track_by_id(trigger_entity_id["trackId"])
        if track is None:
            raise Exception("Cannot find track")
        clip = track.get_clip_by_id(trigger_entity_id["clipId"])
        if clip is None:
            raise Exception("Cannot find clip")
        clip_audio_data_list: ClipAudioDataInjectData = params["clipAudioData"]
        new_midi_track = song.create_track(type=TrackType.MIDI_TRACK, index=song.get_track_index(
            track_id=track.get_id()),
            assign_default_sampler_plugin=True)

        tmp_file = tempfile.NamedTemporaryFile(delete=True, suffix=clip_audio_data_list[0]["audioData"]["format"])
        tmp_file.write(clip_audio_data_list[0]["audioData"]["data"])

        try:
            TranscribeSinging._transcribe_clip(predictor, song,
                                            new_midi_track,
                                            clip,
                                            tmp_file.name,
                                            False,
                                            params["onsetThreshold"],
                                            params["silenceThreshold"])
        except Exception as e:
            print(traceback.format_exc)
        finally:
            tmp_file.close()


    @staticmethod
    def _transcribe_clip(
        predictor,
        song: Song,
        new_midi_track: Track,
        audio_clip: Clip,
        audio_file_path,
        do_separation=False,
        onset_threshold=0.4,
        silence_threshold=0.5,
    ):
        new_clip = new_midi_track.create_midi_clip(
            clip_start_tick=audio_clip.get_clip_start_tick(),
            clip_end_tick=audio_clip.get_clip_end_tick(),
            insert_clip=True
        )
        audio_start_tick = audio_clip.get_audio_clip_data().start_tick  # type:ignore
        audio_start_time = song.tick_to_seconds(audio_start_tick)
        # TODO: Trim the audio so that we only transcribe the visible part.
        test_dataset = SeqDataset(audio_file_path, song_id='1', do_svs=do_separation)

        results = {}
        results = predictor.predict(test_dataset, results=results,
                                    onset_thres=onset_threshold, offset_thres=silence_threshold)

        for notes in results['1']:
            note_start_time_within_audio = notes[0]
            note_start_tick = song.seconds_to_tick(note_start_time_within_audio + audio_start_time)
            note_end_time_within_audio = notes[1]
            note_end_tick = song.seconds_to_tick(note_end_time_within_audio + audio_start_time)
            note_pitch = notes[2]

            new_clip.create_note(
                pitch=note_pitch,
                velocity=100,
                start_tick=note_start_tick,
                end_tick=note_end_tick
            )
        new_clip.adjust_clip_left(clip_start_tick=audio_clip.get_clip_start_tick(), resolve_conflict=False)
        new_clip.adjust_clip_right(clip_end_tick=audio_clip.get_clip_end_tick(), resolve_conflict=False)
