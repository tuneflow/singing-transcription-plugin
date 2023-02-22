from __future__ import annotations

from tuneflow_py import TuneflowPlugin, Song, ReadAPIs, ParamDescriptor, WidgetType, TrackType, ClipType, Note, Track
from typing import Any
import sys
import librosa
import audioread.ffdec  # Use ffmpeg decoder
from data_utils.seq_dataset import SeqDataset
from predictor import EffNetPredictor
import torch

class TranscribeSinging(TuneflowPlugin):
    @staticmethod
    def provider_id():
        return "Hellwz"

    @staticmethod
    def plugin_id():
        return "singing-transcription"

    @staticmethod
    def provider_display_name():
        return {
            "zh": "WZW",
            "en": "WZW"
        }

    @staticmethod
    def plugin_display_name():
        return {
            "zh": "人声转MIDI",
            "en": "Singing Transcription"
        }

    @staticmethod
    def plugin_description():
        return {
            "zh": "将人声歌声转为MIDI旋律",
            "en": "Transcribe singing to MIDI notes"
        }

    @staticmethod
    def allow_reset():
        return False

    def params(self) -> dict[str, ParamDescriptor]:
        return {
            "audio": {
                "displayName": {
                    "zh": '音频',
                    "en": 'Audio',
                },
                "defaultValue": None,
                "widget": {
                    "type": WidgetType.MultiSourceAudioSelector.value,
                    "config": {
                        "allowedSources": ['audioTrack', 'file'],
                    }
                },
            },
            "doSeparation":{
                "displayName": {
                    "zh": '自动分离伴奏',
                    "en": 'Automatic accompaniment separation',
                },
                "defaultValue": False,
                "description": {
                    "zh": '先自动分离音频中的伴奏，再进行人声转录',
                    "en": 'Automatically separate the accompaniment from the audio first, then transcribe the vocals',
                },
                "widget": {
                    "type": WidgetType.Switch.value,
                },
            },
            "onsetThreshold":{
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
                    "config":{
                        "minValue": 0.1,
                        "maxValue": 0.9,
                        "step": 0.1
                    }
                },
            },
            "silenceThreshold":{
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
                    "config":{
                        "minValue": 0.1,
                        "maxValue": 0.9,
                        "step": 0.1
                    }
                },
            }
        }

    def init(self, song: Song, read_apis: ReadAPIs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.predictor = EffNetPredictor(device=device, model_path="models/1005_e_4")

    def run(self, song: Song, params: dict[str, Any], read_apis: ReadAPIs):

        audio = params["audio"]
        # print(audio.keys())
        
        # print(dir(song._proto.tracks))
        # print(dir(new_midi_track.clips))
        # print(new_midi_track.type)

        # new_track_id = params["midi"]
        # new_track = song.get_track_by_id(track_id=new_track_id)
        # new_clip = new_track.create_clip()

        # # createTrack({
        # #         type: TrackType.MIDI_TRACK,
        # #         index: song.getTrackIndex(track.getId()),
        # #         assignDefaultSamplerPlugin: true,
        # #     });

        song.print_all_tracks()

        if audio["sourceType"] == 'audioTrack':
            # print(type(audio["audioInfo"]))
            track_id = audio["audioInfo"]

            track = song.get_track_by_id(track_id=track_id)
            if track is None:
                raise Exception("Track not ready")
            if track.get_type() != TrackType.AUDIO_TRACK:
                raise Exception("Can only transcribe audio tracks")
            
            track_index = song.get_track_index_by_id(track_id=track_id) # selected audio track
            # print(track_index)
            new_midi_track = song.create_track(type=TrackType.MIDI_TRACK)
            
            clip_index = 0
            for clip in track.get_clips():

                # print(dir(track._proto))
                # print(dir(clip._proto))

                if clip.get_type() != ClipType.AUDIO_CLIP:
                    raise Exception("Skip non-audio clip")
                audio_clip_data = clip.get_audio_clip_data()
                if audio_clip_data is None or audio_clip_data.audio_file_path is None:
                    continue
                # print(audio_clip_data.audio_file_path)

                new_clip = new_midi_track.create_clip(
                    type=ClipType.MIDI_CLIP,
                    clip_start_tick=clip.get_clip_start_tick(),
                    clip_end_tick=clip.get_clip_end_tick()
                )

                new_midi_track.insert_clip(index=clip_index, clip=new_clip)

                print(clip.get_clip_start_tick(), clip.get_clip_end_tick())

                clip_index = clip_index + 1

                # self._transcribe_clip(audio_clip_data.audio_file_path, params["doSeparation"], params["onsetThreshold"], params["silenceThreshold"])

        elif audio["sourceType"] == 'file':
            print(type(audio["audioInfo"]))

        # print(sys.getsizeof(audio["audioInfo"]))
        # aro = audioread.ffdec.FFmpegAudioFile(audio["audioInfo"])
        # y, sr = librosa.load(aro)
        # print(y)
        # print(audio["audioInfo"].decode(encoding='UTF-16'))

        # separate = params["separate"]
        # print(separate)

        song.insert_track(index=track_index+1, track=new_midi_track)
        print(track.print_all_clips())
        print(new_midi_track.print_all_clips())
        print(song.print_all_tracks())

    def _transcribe_clip(self, audio_file_path, do_separation=False, onset_threshold=0.4, silence_threshold=0.5):
        test_dataset = SeqDataset(audio_file_path, song_id='1', do_svs=do_separation)

        results = {}
        results = self.predictor.predict(test_dataset, results=results, onset_thres=onset_threshold, offset_thres=silence_threshold)

        print(results['1'])
