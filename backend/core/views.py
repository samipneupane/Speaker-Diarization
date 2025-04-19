from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import os
import shutil
from django.core.files.storage import default_storage

from core.logic.eend_eda.inference_diarization import speaker_diarization_eend
from core.logic.inference_translation import np_speech_text_translation
from core.logic.visualize import diarization_result_base64
from core.serializers import AudioFileSerializer


class SpeakerDiarizationView(CreateAPIView):
    serializer_class = AudioFileSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        audio_dir = 'core/logic/user_input/user_0'
        os.makedirs(audio_dir, exist_ok=True)
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        file_obj = serializer.validated_data['audio']
        model_choice = serializer.validated_data['model']
        spk_choice = serializer.validated_data['spk']

        file_path = os.path.join(audio_dir, file_obj.name)

        try:
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in file_obj.chunks():
                    destination.write(chunk)
            
            if model_choice == "eend-eda":
                model_path = f"core/logic/eend_eda/model/{spk_choice}spk/models"
                if spk_choice == "2":
                    init_epoch = "14-19"
                    spk_qty = 2
                if spk_choice == "3":
                    init_epoch = "14-20"
                    spk_qty = 3
                if spk_choice == "4":
                    init_epoch = "35-40"
                    spk_qty = 4
                    model_path = f"core/logic/eend_eda/model/old/4spkos3"
                if spk_choice == "M":
                    init_epoch = "29-35"
                    spk_qty = 4
                    model_path = f"core/logic/eend_eda/model/old/4spkos4_last"

                # model_path = "core/logic/eend_eda/model/old/4spkos3"
                # init_epoch = "34-40"
                # spk_qty = 4

                # print(model_path, model_choice, init_epoch, spk_qty)
                output = speaker_diarization_eend(audio_dir, model_path, init_epoch, spk_qty)

            if model_choice == "diaper":
                model_path = f"core/logic/diaper/model/{spk_choice}spk/models"
                if spk_choice == "2":
                    init_epoch = "45-50"
                    spk_qty = 2
                if spk_choice == "3":
                    init_epoch = "45-50"
                    spk_qty = 3
                if spk_choice == "4":
                    init_epoch = "40-45"
                    spk_qty = 4
                if spk_choice == "M":
                    init_epoch = "25-30"
                    spk_qty = 4

                # print(model_path, model_choice, init_epoch, spk_qty)

            text_list = np_speech_text_translation(audio_dir)
            for i, spk in enumerate(output):
                spk.append(text_list[i])

            image_base64 = diarization_result_base64(file_path, os.path.join(audio_dir, f'{os.path.splitext(file_obj.name)[0]}.rttm'))

            return Response({'diarization_result': output, 'image': image_base64}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            shutil.rmtree(audio_dir)
            os.makedirs(audio_dir, exist_ok=True)