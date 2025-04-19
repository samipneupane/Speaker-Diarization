from djoser.serializers import UserCreateSerializer, UserSerializer
from rest_framework import serializers


class UserCreateSer(UserCreateSerializer):
    class Meta(UserCreateSerializer.Meta):
        fields = ['id', 'username', 'password', 'email', 'first_name', 'last_name', 'phone_number']
        
        
        
class UserSer(UserSerializer):
    class Meta(UserSerializer.Meta):
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone_number']
        


class AudioFileSerializer(serializers.Serializer):
    MODEL_CHOICES = [
        ('eend-eda', 'EEND-EDA'),
        ('diaper', 'DiaPer'),
    ]

    SPK_CHOICES = [
        ('2', '2 Speakers'),
        ('3', '3 Speakers'),
        ('4', '4 Speakers'),
        ('M', 'Mixed Speakers'),
    ]

    model = serializers.ChoiceField(choices=MODEL_CHOICES)
    spk = serializers.ChoiceField(choices=SPK_CHOICES)
    audio = serializers.FileField()
